"""Veiled clearance: concealed front-edge marks program a route.

Only numpy and the standard library are used. The sampler renders complete scene
descriptors; the solver independently reads the visible pixels. No files, seeds,
training labels, or cached descriptors are consulted by solve().
"""

import hashlib
import json
import numpy as np


HOST_SEEDS = (0, 231975634, 995434905, 487279474, 789264031, 856775692,
              458667262, 680088650, 894003295, 145307459, 730514412,
              716133665, 32608383, 330198923, 843980113, 462451589,
              168787987, 26070418, 872415353, 25601112, 621711112,
              169893502, 575630317, 694707820, 720581982, 866684722,
              542793594, 762766211, 457589979, 670427771, 641896959,
              582970137, 617553219, 668443231, 413648781, 771426978,
              265068589, 532268901, 793614732, 73374489, 397268860,
              256274264)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _counts(rng, a, b):
    # None is a supported convenience; the public defaults are exactly 4/1.
    if a is not None:
        a = _integer(a, "num_train")
        if not 2 <= a <= 10:
            raise ValueError("num_train must be in 2..10")
    if b is not None:
        b = _integer(b, "num_test")
        if not 1 <= b <= 4:
            raise ValueError("num_test must be in 1..4")
    if a is None and b is None:
        b = int(rng.integers(1, 5))
        a = int(rng.integers(b + 1, 11))
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


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


def _components(points):
    todo, result = set(points), []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, group = [first], {first}
        while stack:
            y, x = stack.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _orient(a, direction):
    # Canonical direction is down; 0,1,2,3 yield down,right,up,left.
    return np.rot90(a, direction).copy()


def _marks(box, m, side):
    t, b, l, r = box
    xs = [l + 2 * j if side == -1 else r - 2 * j for j in range(m)]
    return {(y, x) for y in (t, b) for x in xs}


def _grid_key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _shutters(a, bg):
    """Uniform interior components surrounded by a solid contrasting frame."""
    found = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        for group in _components(map(tuple, np.argwhere(a == color))):
            t, b, l, r = _bbox(group)
            if len(group) != (b - t + 1) * (r - l + 1):
                continue
            if min(t, l) < 1 or b >= a.shape[0] - 1 or r >= a.shape[1] - 1:
                continue
            box = (t - 1, b + 1, l - 1, r + 1)
            border = {int(a[p]) for p in _perimeter(box)}
            if len(border) == 1 and not border & {bg, color}:
                found.append((box, border.pop(), color))
    assert found, "no readable shutters"
    assert len({(f, c) for _, f, c in found}) == 1, "inconsistent shutter roles"
    assert len(found) <= 4
    masked = set()
    for box, _, _ in found:
        cells = _area(box)
        assert not cells & masked, "overlapping shutters"
        masked |= cells
    return found, masked


def _decode(input_grid):
    raw = np.asarray(input_grid)
    assert raw.ndim == 2 and np.issubdtype(raw.dtype, np.integer)
    assert all(1 <= d <= 30 for d in raw.shape) and np.all((raw >= 0) & (raw <= 9))
    a = raw.astype(np.int8, copy=True)
    bg = int(a[0, 0])
    assert all(int(a[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    # The only non-background boundary pixel is the inward route seed.
    edge = np.zeros(a.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    seeds = np.argwhere(edge & (a != bg))
    assert seeds.shape == (1, 2), "one boundary seed required"
    y, x = map(int, seeds[0])
    matches = (y == 0, x == 0, y == a.shape[0] - 1, x == a.shape[1] - 1)
    assert sum(matches) == 1
    direction = matches.index(True)
    route = int(a[y, x])
    assert np.count_nonzero(a == route) == 1
    a = np.rot90(a, -direction).copy()
    initial = int(np.argwhere(a == route)[0, 1])
    shutters, masked = _shutters(a, bg)
    frame, fill = shutters[0][1:]
    assert len({bg, route, frame, fill}) == 4
    points = {tuple(map(int, p)) for p in np.argwhere((a != bg) & (a != route))} - masked
    colors = {int(a[p]) for p in points}
    assert len(colors) == 2 and not colors & {bg, route, frame, fill}
    groups = _components(points)
    assert 1 <= len(groups) <= 4
    explanations = []
    for wall in sorted(colors):
        marker = (colors - {wall}).pop()
        decoded = []
        for visible in groups:
            box = _bbox(visible)
            t, b, l, r = box
            if b - t < 3 or r - l < 2:
                break
            full = _perimeter(box)
            if full - masked != visible:
                break
            options = []
            for m in range(1, 5):
                if r - l + 1 < 2 * m + 1:
                    continue
                for side in (-1, 1):
                    marks = _marks(box, m, side)
                    if all(int(a[p]) == (marker if p in marks else wall) for p in visible):
                        options.append((m, side, marks))
            if len(options) != 1:
                break
            m, side, marks = options[0]
            # At least one complete transverse edge witnesses the entire word.
            assert (not masked & {(t, c) for c in range(l, r + 1)} or
                    not masked & {(b, c) for c in range(l, r + 1)})
            n = sum(p in masked for p in marks if p[0] == t)
            rear = sum(p in masked for p in marks if p[0] == b)
            decoded.append({"box": box, "m": m, "side": side, "marks": marks,
                            "n": n, "rear": rear, "hidden": full & masked})
        else:
            explanations.append((wall, marker, sorted(decoded, key=lambda q: q["box"])))
    assert len(explanations) == 1, "rectangle instructions not uniquely recoverable"
    wall, marker, boxes = explanations[0]
    for first, second in zip(boxes, boxes[1:]):
        assert first["box"][1] < second["box"][0], "overlapping forward bands"
    # Each shutter belongs to exactly one loop, and hides exactly one corner.
    for shutter, _, _ in shutters:
        cells = _area(shutter)
        owners = [q for q in boxes if cells & _perimeter(q["box"])]
        assert len(owners) == 1
        t, b, l, r = owners[0]["box"]
        assert sum(p in cells for p in ((t, l), (t, r), (b, l), (b, r))) == 1
    return {"array": a, "direction": direction, "background": bg, "route": route,
            "wall": wall, "marker": marker, "frame": frame, "fill": fill,
            "shutters": shutters, "masked": masked, "initial": initial, "boxes": boxes}


def _trace(scene, mode="correct", strict=True):
    a = scene["array"]
    path = np.zeros(a.shape, dtype=bool)
    current, start, previous = scene["initial"], 0, -1
    hits = []
    for q in scene["boxes"]:
        t, b, l, r = q["box"]
        hit = l <= current <= r or mode == "all_boxes"
        hits.append(hit)
        if not hit:
            previous = b
            continue
        n = q["n"]
        if mode == "total":
            n = q["m"]
        elif mode == "all_hidden":
            n += q["rear"]
        elif mode == "no_occlusion":
            n = 0
        elif mode == "visible":
            n = q["m"] - q["n"]
        d = n + 1
        turn = t - d
        target = r + d if q["side"] == -1 else l - d
        if mode == "marked_end":
            target = l - d if q["side"] == -1 else r + d
        valid = 1 <= turn < a.shape[0] and previous < turn and start <= turn and 0 <= target < a.shape[1]
        if not valid:
            if strict:
                raise AssertionError("infeasible route command")
            return None, hits
        # Stepwise construction differs from the sampler's slice renderer.
        for y in range(start, turn + 1):
            path[y, current] = True
        sign = 1 if target > current else -1
        for x in range(current, target + sign, sign):
            path[turn, x] = True
        current, start, previous = target, turn, b
    for y in range(start, a.shape[0]):
        path[y, current] = True
    footprint = set().union(*(_area(q["box"]) for q in scene["boxes"]))
    collision = any(path[p] for p in footprint)
    if strict:
        assert not collision, "route enters a reconstructed rectangle"
    elif collision:
        return None, hits
    return path, hits


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reveal and normalize all loops; route using concealed front marks only."""
    scene = _decode(input_grid)
    path, _ = _trace(scene)
    out = np.full(scene["array"].shape, scene["background"], dtype=np.int8)
    for q in scene["boxes"]:
        for p in _perimeter(q["box"]):
            out[p] = scene["wall"]
    out[path] = scene["route"]
    return _orient(out, scene["direction"])


def _span(rng, width, current, m, n, side, hit):
    # Three-cell margins leave room for a shutter beyond the marker endpoint.
    choices = []
    for l in range(3, width - 5):
        for r in range(l + 2 * m, min(width - 3, l + 16)):
            intercept = l <= current <= r
            if intercept != hit:
                continue
            target = r + n + 1 if side == -1 else l - n - 1
            if hit and not 0 <= target < width:
                continue
            if not hit and target == current:
                continue
            choices.append((l, r, target if hit else current))
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _sample_pair(rng, palette, direction, requests, sides, require_partial=False):
    # A bounded rejection sampler; requests are (hit, front-hidden n, mask kind).
    for _ in range(400):
        width = int(rng.integers(22, 31))
        current = int(rng.integers(3, width - 3))
        initial, boxes = current, []
        for (hit, n, kind), forced_side in zip(requests, sides):
            lower = n + 1 if require_partial and hit and n else max(1, n)
            m = int(rng.integers(lower, 5))
            # Every positive command is at least sometimes a partial word.
            if n and m == n and n < 4 and rng.random() < 0.8:
                m += 1
            side = forced_side if forced_side else int(rng.choice((-1, 1)))
            span = _span(rng, width, current, m, n, side, hit)
            if span is None:
                break
            l, r, target = span
            boxes.append({"m": m, "n": n, "side": side, "hit": hit,
                          "kind": kind, "l": l, "r": r, "target": target})
            current = target
        else:
            # All geometry fits a maximum 30-cell forward extent. Positive
            # commands consume room according to their actual hidden count.
            minimum = 1 + sum(4 + (q["n"] + 3 if q["hit"] else 3) for q in boxes)
            if minimum > 30:
                continue
            slack = int(rng.integers(0, 31 - minimum))
            allocations = rng.multinomial(slack, np.full(2 * len(boxes) + 1, 1 / (2 * len(boxes) + 1)))
            previous = -2
            for i, q in enumerate(boxes):
                gap = q["n"] + 4 if q["hit"] else 4
                top = previous + gap + int(allocations[2 * i])
                bottom = top + 3 + int(allocations[2 * i + 1])
                q["box"] = (top, bottom, q["l"], q["r"])
                previous = bottom
            height = previous + 3 + int(allocations[-1])
            if height > 30:
                continue
            bg, wall, marker, frame, fill, route = palette
            a = np.full((height, width), bg, dtype=np.int8)
            out = a.copy()
            for q in boxes:
                for p in _perimeter(q["box"]):
                    a[p] = out[p] = wall
                for p in _marks(q["box"], q["m"], q["side"]):
                    a[p] = marker
            shutters = []
            for q in boxes:
                if q["kind"] == "none":
                    continue
                t, b, l, r = q["box"]
                hidden = q["n"] if q["kind"] == "front" else int(rng.integers(1, q["m"] + 1))
                cut = 2 * (hidden - 1) + int(rng.integers(2))
                extension_min = 2 if cut == 0 else 1
                if q["side"] == -1:
                    x0, x1 = l - int(rng.integers(extension_min, min(l - 1, 3) + 1)), l + cut
                else:
                    x0, x1 = r - cut, r + int(rng.integers(extension_min, min(3, width - 2 - r) + 1))
                depth = int(rng.integers(1, 3))
                box = ((t - 1, t + depth, x0, x1) if q["kind"] == "front" else
                       (b - depth, b + 1, x0, x1))
                shutters.append(box)
                y0, y1, x0, x1 = box
                a[y0:y1 + 1, x0:x1 + 1] = fill
                for p in _perimeter(box):
                    a[p] = frame
            a[0, initial] = route
            # Ground truth uses sampled full boxes, never the pixel decoder.
            current, start = initial, 0
            for q in boxes:
                if not q["hit"]:
                    continue
                top = q["box"][0]
                turn = top - q["n"] - 1
                target = q["target"]
                out[start:turn + 1, current] = route
                out[turn, min(current, target):max(current, target) + 1] = route
                current, start = target, turn
            out[start:, current] = route
            pair = {"input": _orient(a, direction), "output": _orient(out, direction)}
            try:
                scene = _decode(pair["input"])
                _, hits = _trace(scene)
                assert hits == [q["hit"] for q in boxes]
                assert [q["n"] for q in scene["boxes"]] == [q["n"] for q in boxes]
                assert np.array_equal(solve(pair["input"]), pair["output"])
            except AssertionError:
                continue
            return pair
    raise AssertionError("bounded scene sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    maximum = int(rng.integers(2, 4))
    directions = list(map(int, rng.permutation(4)))
    palette = tuple(map(int, rng.permutation(10)[:6]))
    recolor = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                n = 1 if i == 0 else maximum if i == 1 else int(rng.choice((1, maximum)))
                requests = [(True, n, "front"), (True, 0, "rear")]
                if i == 0 or rng.random() < 0.55:
                    requests.append((False, int(rng.choice((1, maximum))), "front"))
                if i >= 2:
                    size = int(rng.choice((1, 2, 3, 4), p=(0.2, 0.3, 0.35, 0.15)))
                    if size == 1:
                        requests = requests[:1]
                    elif size == 4:
                        # Four boxes require the compact unary-one case.
                        requests = [(True, 1, "front"), (True, 0, "rear"),
                                    (True, 0, "none"), (False, 0, "none")]
                forced = -1 if i == 0 else 1 if i == 1 else int(rng.choice((-1, 1)))
                sides = [forced] + [0] * (len(requests) - 1)
                if i < 2:
                    # Repeated and alternating transitions are both visible
                    # before tests combine multiple concealed commands.
                    sides[1] = forced if i == 0 else -forced
                direction = directions[i % 3]
            else:
                ns = list(map(int, rng.choice((1, maximum), 2)))
                requests = [(True, n, "front") for n in ns]
                if rng.random() < 0.65:
                    requests.append((bool(rng.integers(2)), 0, str(rng.choice(("rear", "none")))))
                sides = [0] * len(requests)
                direction = directions[3]
            order = list(map(int, rng.permutation(len(requests))))
            requests = [requests[j] for j in order]
            sides = [sides[j] for j in order]
            for _ in range(16):
                colors = tuple(map(int, rng.permutation(10)[:6])) if recolor else palette
                pair = _sample_pair(rng, colors, direction, requests, sides,
                                    require_partial=split == "train" and i == 0)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct input sampling exhausted")
    # Presentation order carries no role information.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    scene = _decode(a)
    path, hits = _trace(scene)
    boxes = scene["boxes"]
    alt = {}
    for mode in ("total", "all_hidden", "no_occlusion", "visible", "marked_end", "all_boxes"):
        alternative, _ = _trace(scene, mode, strict=False)
        alt[mode] = alternative is None or not np.array_equal(alternative, path)
    return {"direction": scene["direction"], "shape": tuple(a.shape),
            "boxes": len(boxes), "shutters": len(scene["shutters"]),
            "hits": sum(hits), "misses": len(boxes) - sum(hits),
            "positive_hits": sum(hit and q["n"] > 0 for hit, q in zip(hits, boxes)),
            "counts": tuple(q["n"] for hit, q in zip(hits, boxes) if hit),
            "words": tuple(q["m"] for q in boxes),
            "sides": tuple(q["side"] for hit, q in zip(hits, boxes) if hit),
            "partial": any(hit and 0 < q["n"] < q["m"] for hit, q in zip(hits, boxes)),
            "rear_zero": any(hit and q["n"] == 0 and q["rear"] > 0 for hit, q in zip(hits, boxes)),
            "masked_decoy": any(not hit and q["n"] > 0 for hit, q in zip(hits, boxes)),
            "palette": tuple(scene[k] for k in ("background", "wall", "marker", "frame", "fill", "route")),
            "layout": tuple((q["box"], q["m"], q["n"], q["side"]) for q in boxes),
            "alternatives_differ": alt}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grids = {k: np.asarray(v) for k, v in pair.items()}
            for grid in grids.values():
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = grids["input"], grids["output"]
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert np.array_equal(solve(inp), out), "incorrect revealed route"
            f = _features(inp)
            features[split].append(f)
            # Shutter colors disappear; all visible and concealed instructions
            # become permanent walls; no extra colors are invented.
            bg, wall, marker, frame, fill, route = f["palette"]
            assert set(map(int, np.unique(out))) == {bg, wall, route}
            assert np.all(out[inp == wall] == wall)
            assert np.all(out[inp == marker] == wall)
            assert np.count_nonzero(out == route) > 1
    train, test = features["train"], features["test"]
    assert all(f["positive_hits"] == 1 for f in train)
    assert all(f["positive_hits"] == 2 for f in test)
    counts = {n for f in train for n in f["counts"]}
    assert 0 in counts and 1 in counts and max(counts) >= 2
    assert {n for f in test for n in f["counts"]} <= counts
    assert {side for f in train for side in f["sides"]} == {-1, 1}
    assert any(f["partial"] for f in train)
    assert any(f["rear_zero"] for f in train)
    assert any(f["masked_decoy"] for f in train)
    transitions = {(a, b) for f in train for a, b in zip(f["sides"], f["sides"][1:])}
    assert any(a == b for a, b in transitions) and any(a != b for a, b in transitions)
    for alternative in ("total", "all_hidden", "no_occlusion", "visible", "marked_end", "all_boxes"):
        assert any(f["alternatives_differ"][alternative] for f in train), alternative
    assert not {f["direction"] for f in train} & {f["direction"] for f in test}


def self_test():
    """42 seeds, all explicit counts, counterexamples and invariance; no files."""
    shapes, layouts, palettes, directions, counts, words, transitions = (set() for _ in range(7))
    digests, summary = set(), {"seeds": 42, "pairs": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed))
        digests.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                shapes.add(f["shape"])
                layouts.add(f["layout"])
                palettes.add(f["palette"])
                directions.add((split, f["direction"]))
                counts.update(f["counts"])
                words.update(f["words"])
                transitions.update(zip(f["sides"], f["sides"][1:]))
                summary["pairs"] += 1
    assert len(digests) == 42 and len(shapes) >= 35 and len(layouts) >= 180
    assert len(palettes) >= 30 and len(directions) == 8
    assert counts >= {0, 1, 2, 3} and words == {1, 2, 3, 4}
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(100 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((None, None), (None, 4), (2, None), (None, 1)):
        validate_generated_puzzle(generate_puzzle(91, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # Literal hand-derived scene: top row 5, left-end word M W M W W W W,
    # first marker under shutter, n=1, d=2, turn row 3, exit column 12.
    a = np.zeros((15, 18), dtype=np.int8)
    box = (5, 9, 4, 10)
    for p in _perimeter(box):
        a[p] = 7
    for p in _marks(box, 2, -1):
        a[p] = 2
    shutter = (4, 6, 2, 4)
    a[4:7, 2:5] = 8
    for p in _perimeter(shutter):
        a[p] = 6
    a[0, 7] = 3
    expected = np.zeros_like(a)
    for p in _perimeter(box):
        expected[p] = 7
    expected[:4, 7] = expected[3, 7:13] = expected[3:, 12] = 3
    assert np.array_equal(solve(a), expected)
    # Move the shutter to the rear corner: same loop and word, n=0, d=1.
    rear = np.zeros_like(a)
    for p in _perimeter(box):
        rear[p] = 7
    for p in _marks(box, 2, -1):
        rear[p] = 2
    rear[8:11, 2:5] = 8
    for p in _perimeter((8, 10, 2, 4)):
        rear[p] = 6
    rear[0, 7] = 3
    expected_rear = np.zeros_like(a)
    for p in _perimeter(box):
        expected_rear[p] = 7
    expected_rear[:5, 7] = expected_rear[4, 7:12] = expected_rear[4:, 11] = 3
    assert np.array_equal(solve(rear), expected_rear)
    permutation = np.array([9, 2, 4, 0, 6, 8, 1, 5, 7, 3])
    for direction in range(4):
        for flip in (False, True):
            x, y = (a[:, ::-1], expected[:, ::-1]) if flip else (a, expected)
            assert np.array_equal(solve(permutation[_orient(x, direction)]),
                                  permutation[_orient(y, direction)])
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    summary.update({"input_shapes": len(shapes), "layouts": len(layouts),
                    "palettes": len(palettes), "directions_by_split": len(directions),
                    "hidden_counts": sorted(counts), "word_lengths": sorted(words),
                    "side_transitions": sorted(transitions), "distinct_puzzles": len(digests)})
    return summary


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
