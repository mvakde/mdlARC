"""Keyed tally inlays: count in a frame, enumerate around pegs, transport the order.

Only NumPy and the Python standard library are used. The solver reads pixels;
no generation parameters, examples, random state, or cached solutions are used.
"""
from __future__ import annotations

import numpy as np


def _transform(a, operation):
    return np.rot90(a if operation < 4 else np.fliplr(a), operation % 4).copy()


def _grid_key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        visited[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells) + 1, max(c for r, c in cells) + 1)


def _rim(a, color):
    return bool(np.all(a[0] == color) and np.all(a[-1] == color)
                and np.all(a[:, 0] == color) and np.all(a[:, -1] == color))


def _stamp(mask, count, payload, carrier):
    """The ordering is over the source's non-key cells, not a bounding-box prefix."""
    a = np.full(mask.shape, carrier, dtype=np.int8)
    available = np.flatnonzero(~mask.ravel())
    if not 0 <= count <= len(available):
        raise ValueError("count exceeds stencil capacity")
    if count:
        a.flat[available[:count]] = payload
    return a


def _read_input(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any(g < 0) or np.any(g > 9)):
        raise ValueError("invalid ARC grid")
    g = g.astype(np.int8)
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    bg = int(border[0])
    if not np.all(border == bg):
        raise ValueError("the canvas must have a uniform background border")
    objects = []
    for color in np.unique(g):
        if color != bg:
            objects.extend((len(cells), int(color), cells) for cells in _components(g == color))
    if not objects:
        raise ValueError("no board")
    objects.sort(key=lambda x: x[0], reverse=True)
    if len(objects) > 1 and objects[0][0] == objects[1][0]:
        raise ValueError("ambiguous board component")
    _, carrier, main = objects[0]
    top, left, bottom, right = _bounds(main)
    board = g[top:bottom, left:right].copy()
    if (min(board.shape) < 9 or not _rim(board, carrier)
            or not set(map(int, np.unique(board))) <= {bg, carrier}
            or not np.any(board == bg)):
        raise ValueError("no valid carrier board")
    outside_carrier = g == carrier
    outside_carrier[top:bottom, left:right] = False
    rectangles = set()
    for cells in _components(outside_carrier):
        r, c, b, d = _bounds(cells)
        frame = g[r:b, c:d]
        if min(frame.shape) >= 4 and _rim(frame, carrier):
            rectangles.add((r, c, b, d))
    sources, covered_carrier = [], np.zeros(g.shape, dtype=bool)
    occupied = np.zeros(g.shape, dtype=bool)
    occupied[top:bottom, left:right] = True
    for r, c, b, d in sorted(rectangles):
        if np.any(occupied[r:b, c:d]):
            raise ValueError("nested or overlapping source frames")
        frame = g[r:b, c:d].copy()
        mask = frame == carrier
        interior = frame[1:-1, 1:-1]
        payload_cells = (interior != bg) & (interior != carrier)
        colors = np.unique(interior[payload_cells])
        if len(colors) > 1:
            raise ValueError("a source interior has multiple payload colors")
        n = int(payload_cells.sum())
        payload = int(colors[0]) if n else None
        if n > 9:
            raise ValueError("enclosed count exceeds nine")
        sources.append({"origin": (r, c), "array": frame, "mask": mask,
                        "count": n, "payload": payload,
                        "capacity": int((~mask).sum()),
                        "stamp": _stamp(mask, n, payload, carrier)})
        covered_carrier[r:b, c:d] |= mask
        occupied[r:b, c:d] = True
    if not sources or np.any(outside_carrier & ~covered_carrier):
        raise ValueError("unrecognized carrier outside board")
    exterior = (~occupied) & (g != bg)
    if np.any(g[exterior] == carrier):
        raise ValueError("exterior carrier")
    return {"grid": g, "background": bg, "carrier": carrier,
            "board": board, "board_origin": (top, left),
            "sources": sources, "exterior": exterior}


def _reconstruct(input_grid):
    info = _read_input(input_grid)
    board, sources = info["board"], info["sources"]
    bg, carrier = info["background"], info["carrier"]
    bh, bw = board.shape
    holes = board == bg
    all_holes = sum(1 << (int(r) * bw + int(c)) for r, c in np.argwhere(holes))
    window_lookup = {}
    candidate_lists = []
    for source in sources:
        candidates, seen = [], set()
        for operation in range(8):
            mask = _transform(source["mask"], operation)
            stamp = _transform(source["stamp"], operation)
            variant_key = mask.shape, mask.tobytes(), stamp.tobytes()
            if variant_key in seen:
                continue
            seen.add(variant_key)
            h, w = mask.shape
            if (h, w) not in window_lookup:
                lookup = {}
                # Intact board perimeter forbids windows outside these bounds.
                for r in range(1, bh - h):
                    for c in range(1, bw - w):
                        key = holes[r:r + h, c:c + w].tobytes()
                        lookup.setdefault(key, []).append((r, c))
                window_lookup[h, w] = lookup
            for r, c in window_lookup[h, w].get(mask.tobytes(), ()):
                occupied = sum(1 << ((r + y) * bw + c + x)
                               for y in range(h) for x in range(w))
                covered = sum(1 << ((r + int(y)) * bw + c + int(x))
                              for y, x in np.argwhere(mask))
                candidates.append({"r": r, "c": c, "mask": mask, "stamp": stamp,
                                   "occupied": occupied, "covered": covered,
                                   "operation": operation})
        if not candidates:
            raise ValueError("a source has no matching negative impression")
        candidate_lists.append(candidates)
    order = sorted(range(len(sources)), key=lambda i: len(candidate_lists[i]))
    outputs, assignment = {}, {}

    def search(index, occupied, covered):
        if len(outputs) >= 2:
            return
        if index == len(order):
            if covered == all_holes:
                out = np.full(board.shape, carrier, dtype=np.int8)
                for candidate in assignment.values():
                    r, c = candidate["r"], candidate["c"]
                    h, w = candidate["stamp"].shape
                    out[r:r + h, c:c + w] = candidate["stamp"]
                outputs.setdefault(out.tobytes(), (out, dict(assignment)))
            return
        i = order[index]
        for candidate in candidate_lists[i]:
            if not occupied & candidate["occupied"]:
                assignment[i] = candidate
                search(index + 1, occupied | candidate["occupied"], covered | candidate["covered"])
                del assignment[i]
                if len(outputs) >= 2:
                    return

    search(0, 0, 0)
    if len(outputs) != 1:
        raise ValueError("expected one distinct completed board")
    out, assignment = next(iter(outputs.values()))
    info.update(output=out, assignment=assignment)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the unique completion entirely from input geometry and colors."""
    return _reconstruct(input_grid)["output"].copy()


def _separated(a, b, gap=0):
    r, c, h, w = a
    rr, cc, hh, ww = b
    return r + h + gap <= rr or rr + hh + gap <= r or c + w + gap <= cc or cc + ww + gap <= c


def _orbit(mask):
    return min((a.shape, a.tobytes()) for a in (_transform(mask, k) for k in range(8)))


def _orientation_count(mask):
    return len({(_transform(mask, k).shape, _transform(mask, k).tobytes()) for k in range(8)})


def _source(rng, bg, carrier, payload, mode="partial", early=False, forbidden_count=None):
    for _ in range(500):
        if early:
            h, w = ((5, 5), (5, 6), (6, 5), (6, 6))[int(rng.integers(4))]
        else:
            h, w = ((4, 4), (4, 5), (5, 4), (5, 5), (4, 6), (6, 4),
                    (5, 6), (6, 5), (6, 6), (5, 7), (7, 5), (7, 7))[int(rng.integers(12))]
        inner_h, inner_w = h - 2, w - 2
        area = inner_h * inner_w
        peg_count = int(rng.integers(1, min(5, area - 2) + 1))
        pegs = np.zeros((inner_h, inner_w), dtype=bool)
        # Boundary tallies allow visibly symmetric key patterns as well.
        if mode in ("zero", "full") and rng.random() < 0.55 and inner_h % 2 and inner_w % 2:
            pegs[inner_h // 2, inner_w // 2] = True
            if rng.random() < 0.35 and inner_h >= 3 and inner_w >= 3:
                pegs[inner_h // 2 - 1, inner_w // 2] = True
                pegs[inner_h // 2 + 1, inner_w // 2] = True
                pegs[inner_h // 2, inner_w // 2 - 1] = True
                pegs[inner_h // 2, inner_w // 2 + 1] = True
        else:
            if early:
                pegs.flat[0] = True
                if peg_count > 1:
                    pegs.flat[rng.choice(np.arange(1, area), peg_count - 1, replace=False)] = True
            else:
                pegs.flat[rng.choice(area, peg_count, replace=False)] = True
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = pegs
        capacity = int((~mask).sum())
        if capacity < 2:
            continue
        if mode == "partial":
            if _orientation_count(mask) != 8:
                continue
            low = max(4, int((~pegs[0]).sum()) + 1) if early else 1
            high = min(9, capacity - 1)
            choices = [n for n in range(low, high + 1) if n != forbidden_count]
            if not choices or (early and (capacity < 6 or not np.any(~pegs[0]))):
                continue
            n = int(rng.choice(choices))
        elif mode == "zero":
            n = 0
        elif mode == "full":
            if capacity > 9:
                continue
            n = capacity
        else:
            raise ValueError("unknown source mode")
        frame = np.where(mask, carrier, bg).astype(np.int8)
        free = np.flatnonzero(~mask.ravel())
        canonical = set(map(int, free[:n]))
        for _ in range(50):
            chosen = np.atleast_1d(rng.choice(free, n, replace=False))
            if mode == "partial" and set(map(int, chosen)) == canonical:
                continue
            if early:
                scattered = np.zeros(mask.shape, dtype=bool)
                scattered.flat[chosen] = True
                if not (np.any(scattered[1:] & scattered[:-1])
                        or np.any(scattered[:, 1:] & scattered[:, :-1])):
                    continue
            break
        else:
            continue
        frame.flat[chosen] = payload
        return {"array": frame, "mask": mask, "count": n, "payload": payload,
                "capacity": capacity, "stamp": _stamp(mask, n, payload, carrier)}
    raise RuntimeError("unable to sample a source frame")


def _pack_board(rng, targets):
    count = len(targets)
    minimum = (9, 9, 11, 14)[count - 1]
    maximum = (17, 19, 20, 20)[count - 1]
    for _ in range(160):
        bh, bw = map(int, rng.integers(minimum, maximum + 1, 2))
        if sum(t.size for t in targets) > (bh - 2) * (bw - 2):
            continue
        order = sorted(range(count), key=lambda i: targets[i].size, reverse=True)
        chosen = {}
        for i in order:
            h, w = targets[i].shape
            positions = [(r, c) for r in range(1, bh - h) for c in range(1, bw - w)
                         if all(_separated((r, c, h, w), box) for box in chosen.values())]
            if not positions:
                break
            if chosen and rng.random() < 0.35:
                adjacent = [(r, c) for r, c in positions if any(
                    (r == rr and (c + w == cc or cc + ww == c))
                    or (c == cc and (r + h == rr or rr + hh == r))
                    for rr, cc, hh, ww in chosen.values())]
                if adjacent:
                    positions = adjacent
            r, c = positions[int(rng.integers(len(positions)))]
            chosen[i] = (r, c, h, w)
        if len(chosen) == count:
            return (bh, bw), [chosen[i][:2] for i in range(count)]
    raise RuntimeError("unable to pack target windows")


def _pack_canvas(rng, board, sources, bg):
    bh, bw = board.shape
    count = len(sources)
    max_h = max(s["array"].shape[0] for s in sources)
    max_w = max(s["array"].shape[1] for s in sources)
    for _ in range(140):
        ch = int(rng.integers(max(bh + max_h + 3, 17 + 2 * count), 31))
        cw = int(rng.integers(max(bw + max_w + 3, 17 + 2 * count), 31))
        if rng.random() < 0.75:
            br = int(rng.choice((1, ch - bh - 1)))
            bc = int(rng.choice((1, cw - bw - 1)))
        else:
            br = int(rng.integers(1, ch - bh))
            bc = int(rng.integers(1, cw - bw))
        boxes = [(br, bc, bh, bw)]
        positions = {}
        for i in sorted(range(count), key=lambda i: sources[i]["array"].size, reverse=True):
            h, w = sources[i]["array"].shape
            slots = [(r, c) for r in range(1, ch - h) for c in range(1, cw - w)
                     if all(_separated((r, c, h, w), box, 1) for box in boxes)]
            if not slots:
                break
            r, c = slots[int(rng.integers(len(slots)))]
            positions[i] = (r, c)
            boxes.append((r, c, h, w))
        if len(positions) != count:
            continue
        g = np.full((ch, cw), bg, dtype=np.int8)
        occupied = np.zeros(g.shape, dtype=bool)
        g[br:br + bh, bc:bc + bw] = board
        occupied[br:br + bh, bc:bc + bw] = True
        for i, (r, c) in positions.items():
            frame = sources[i]["array"]
            h, w = frame.shape
            g[r:r + h, c:c + w] = frame
            occupied[r:r + h, c:c + w] = True
        occupied[0] = occupied[-1] = True
        occupied[:, 0] = occupied[:, -1] = True
        return g, occupied
    raise RuntimeError("unable to scatter source frames")


def _ablation(info, kind):
    out = np.full(info["board"].shape, info["carrier"], dtype=np.int8)
    for i, s in enumerate(info["sources"]):
        a = info["assignment"][i]
        if kind == "copy":
            stamp = np.full(s["array"].shape, info["carrier"], dtype=np.int8)
            if s["count"]:
                stamp[s["array"] == s["payload"]] = s["payload"]
            stamp = _transform(stamp, a["operation"])
        elif kind == "destination":
            stamp = _stamp(a["mask"], s["count"], s["payload"], info["carrier"])
        elif kind == "ignore_pegs":
            stamp = np.full(s["mask"].shape, info["carrier"], dtype=np.int8)
            inner = stamp[1:-1, 1:-1].copy()
            inner.flat[:s["count"]] = s["payload"] if s["count"] else info["carrier"]
            inner[s["mask"][1:-1, 1:-1]] = info["carrier"]
            stamp[1:-1, 1:-1] = inner
            stamp = _transform(stamp, a["operation"])
        else:
            raise ValueError("unknown ablation")
        r, c = a["r"], a["c"]
        h, w = stamp.shape
        out[r:r + h, c:c + w] = stamp
    return out


def _make_pair(rng, colors, count, demonstration=None):
    bg, carrier, pool = colors
    for _ in range(100):
        sources, operations, used_orbits = [], [], set()
        for i in range(count):
            payload = int(rng.choice(pool))
            if demonstration is not None:
                mode = "partial"
            else:
                mode = str(rng.choice(("partial", "zero", "full"), p=(0.70, 0.14, 0.16)))
            for _ in range(40):
                s = _source(rng, bg, carrier, payload, mode,
                            early=(demonstration == "first"),
                            forbidden_count=(sources[0]["count"] if demonstration == "second" and i == 1 else None))
                if demonstration == "second" and i == 0 and s["count"] >= 9:
                    continue
                orbit = _orbit(s["mask"])
                if orbit not in used_orbits:
                    used_orbits.add(orbit)
                    break
            else:
                break
            operation = (0 if demonstration == "first" else
                         int(rng.integers(1, 4)) if demonstration == "second" and i == 0 else
                         int(rng.integers(4, 8)) if demonstration == "second" and i == 1 else
                         int(rng.integers(8)))
            sources.append(s)
            operations.append(operation)
        if len(sources) != count:
            continue
        target_masks = [_transform(s["mask"], op) for s, op in zip(sources, operations)]
        try:
            shape, windows = _pack_board(rng, target_masks)
        except RuntimeError:
            continue
        board = np.full(shape, carrier, dtype=np.int8)
        expected = board.copy()
        for s, mask, op, (r, c) in zip(sources, target_masks, operations, windows):
            h, w = mask.shape
            board[r:r + h, c:c + w][mask] = bg
            expected[r:r + h, c:c + w] = _transform(s["stamp"], op)
        try:
            g, occupied = _pack_canvas(rng, board, sources, bg)
        except RuntimeError:
            continue
        slots = np.argwhere(~occupied)
        exterior_count = int(rng.integers(1 if demonstration == "second" else 0, 10))
        if len(slots) < exterior_count:
            continue
        for j, k in enumerate(np.atleast_1d(rng.choice(len(slots), exterior_count, replace=False))):
            r, c = slots[int(k)]
            g[r, c] = sources[0]["payload"] if j == 0 else int(rng.choice(pool))
        try:
            info = _reconstruct(g)
        except ValueError:
            continue
        if not np.array_equal(info["output"], expected):
            raise AssertionError("construction and reconstruction disagree")
        if demonstration == "first":
            if (np.array_equal(_ablation(info, "copy"), expected)
                    or np.array_equal(_ablation(info, "ignore_pegs"), expected)):
                continue
        if demonstration == "second":
            # Both local examples must visibly transport the reading order.
            if any(np.array_equal(a["stamp"], _stamp(a["mask"], s["count"], s["payload"], carrier))
                   for i, s in enumerate(info["sources"]) for a in (info["assignment"][i],)):
                continue
        return {"input": g, "output": expected}
    raise RuntimeError("unable to construct an unambiguous pair")


def _check_counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    stable = bool(rng.random() < 0.55)
    shared = (palette[0], palette[1], palette[2:2 + int(rng.integers(1, 5))])
    train_counts = [1, 2] + [int(rng.integers(1, 4)) for _ in range(int(num_train) - 2)]
    test_counts = [int(rng.integers(max(train_counts) + 1, 5)) for _ in range(int(num_test))]
    puzzle, used = {"train": [], "test": []}, set()
    for index, count in enumerate(train_counts + test_counts):
        split = "train" if index < num_train else "test"
        demonstration = "first" if index == 0 else "second" if index == 1 else None
        for _ in range(30):
            if stable:
                colors = shared
            else:
                order = tuple(map(int, rng.permutation(10)))
                colors = (order[0], order[1], order[2:2 + int(rng.integers(1, 5))])
            pair = _make_pair(rng, colors, count, demonstration)
            key = _grid_key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("unable to obtain distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                a = np.asarray(grid)
                assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
                assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
            g, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            assert _grid_key(g) not in seen
            seen.add(_grid_key(g))
            info = _reconstruct(g)
            assert np.array_equal(out, info["output"])
            board, bg, carrier = info["board"], info["background"], info["carrier"]
            assert all(9 <= n <= 20 for n in board.shape)
            assert 1 <= len(info["sources"]) <= 4
            assert not np.any(out == bg) and np.all(out[board == bg] == carrier)
            br, bc = info["board_origin"]
            boxes = [(br, bc, *board.shape)]
            counts_by_color = {}
            occupied, covered = 0, 0
            for i, s in enumerate(info["sources"]):
                h, w = s["mask"].shape
                assert 4 <= h <= 7 and 4 <= w <= 7
                assert 1 <= int(s["mask"][1:-1, 1:-1].sum()) <= 5
                assert s["capacity"] >= 2 and 0 <= s["count"] <= min(9, s["capacity"])
                box = (*s["origin"], h, w)
                assert all(_separated(box, other, 1) for other in boxes)
                boxes.append(box)
                a = info["assignment"][i]
                assert not occupied & a["occupied"]
                occupied |= a["occupied"]
                covered |= a["covered"]
                r, c = a["r"], a["c"]
                hh, ww = a["mask"].shape
                assert np.array_equal(board[r:r + hh, c:c + ww] == bg, a["mask"])
                if s["count"]:
                    color = s["payload"]
                    counts_by_color[color] = counts_by_color.get(color, 0) + s["count"]
            all_holes = sum(1 << (int(r) * board.shape[1] + int(c)) for r, c in np.argwhere(board == bg))
            assert covered == all_holes
            assert set(map(int, np.unique(out))) == {carrier} | set(counts_by_color)
            for color, count in counts_by_color.items():
                assert int(np.count_nonzero(out == color)) == count
            # A useful invariance check independent of exterior placement/color.
            cleaned = g.copy()
            cleaned[info["exterior"]] = bg
            assert np.array_equal(solve(cleaned), out)
            analyses[split].append(info)
    first, second = analyses["train"][:2]
    assert len(first["sources"]) == 1 and len(second["sources"]) == 2
    assert not np.array_equal(_ablation(first, "copy"), first["output"])
    assert not np.array_equal(_ablation(first, "ignore_pegs"), first["output"])
    source = first["sources"][0]
    assert 0 < source["count"] < source["capacity"]
    assert len(_components(source["array"] == source["payload"])) < source["count"]
    painted_rows = np.flatnonzero(np.any(source["stamp"] != first["carrier"], axis=1))
    assert len(painted_rows) >= 2
    assert int(second["exterior"].sum()) >= 1
    assert len({s["count"] for s in second["sources"]}) == 2
    rotation = reflection = False
    for i, s in enumerate(second["sources"]):
        a = second["assignment"][i]
        assert 0 < s["count"] < s["capacity"]
        assert not np.array_equal(a["stamp"], _stamp(a["mask"], s["count"], s["payload"], second["carrier"]))
        is_rotation = any(np.array_equal(_transform(s["mask"], k), a["mask"]) for k in range(1, 4))
        is_reflection = not any(np.array_equal(_transform(s["mask"], k), a["mask"]) for k in range(4))
        rotation |= is_rotation
        reflection |= is_reflection
    assert rotation and reflection
    counter_source = next((s for s in second["sources"]
                           if s["count"] < min(9, s["capacity"])
                           and np.any(second["grid"][second["exterior"]] == s["payload"])), None)
    assert counter_source is not None
    # Move one distractor through the boundary. Same total payload, +1 tally.
    moved = second["grid"].copy()
    color = counter_source["payload"]
    er, ec = np.argwhere(second["exterior"] & (moved == color))[0]
    moved[er, ec] = second["background"]
    fr, fc = counter_source["origin"]
    yr, xc = np.argwhere(counter_source["array"] == second["background"])[0]
    moved[fr + yr, fc + xc] = color
    new_output = solve(moved)
    assert int(np.count_nonzero(new_output == color)) == int(np.count_nonzero(second["output"] == color)) + 1
    assert not np.array_equal(new_output, second["output"])
    train_max = max(len(x["sources"]) for x in analyses["train"])
    assert train_max <= 3 and all(len(x["sources"]) > train_max for x in analyses["test"])


def _puzzle_key(p):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in p[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded validation and structural measurements over 42 fixed seeds."""
    shapes, frame_shapes, layouts, masks, palettes, counts, inlay_counts = (set() for _ in range(7))
    signatures, regimes, peg_counts = set(), set(), set()
    endpoints = {"zero": 0, "full": 0, "symmetric": 0}
    exterior_counts, operations = set(), set()
    adjacent = islands = repeated = multiside = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_key(p))
        roles = set()
        for split in ("train", "test"):
            for pair in p[split]:
                info = _reconstruct(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                bg, carrier = info["background"], info["carrier"]
                roles.add((bg, carrier))
                shapes.add(info["board"].shape)
                layouts.add((info["board"].shape, tuple(sorted((a["r"], a["c"], a["mask"].shape)
                                                             for a in info["assignment"].values()))))
                palettes.add((bg, carrier))
                inlay_counts.add(len(info["sources"]))
                exterior_counts.add(int(info["exterior"].sum()))
                payloads, sides, boxes = [], set(), []
                br, bc = info["board_origin"]
                bh, bw = info["board"].shape
                for i, s in enumerate(info["sources"]):
                    frame_shapes.add(s["mask"].shape)
                    masks.add(_orbit(s["mask"]))
                    counts.add(s["count"])
                    peg_counts.add(int(s["mask"][1:-1, 1:-1].sum()))
                    endpoints["zero"] += s["count"] == 0
                    endpoints["full"] += s["count"] == s["capacity"]
                    endpoints["symmetric"] += _orientation_count(s["mask"]) < 8
                    a = info["assignment"][i]
                    operations.add(a["operation"])
                    boxes.append((a["r"], a["c"], *a["mask"].shape))
                    if s["count"]:
                        payloads.append(s["payload"])
                    r, c = s["origin"]
                    h, w = s["mask"].shape
                    if r + h < br:
                        sides.add("top")
                    if r > br + bh:
                        sides.add("bottom")
                    if c + w < bc:
                        sides.add("left")
                    if c > bc + bw:
                        sides.add("right")
                repeated += len(set(payloads)) < len(payloads)
                multiside += len(sides) >= 2
                adjacent += any((r == rr and (c + w == cc or cc + ww == c))
                                or (c == cc and (r + h == rr or rr + hh == r))
                                for j, (r, c, h, w) in enumerate(boxes)
                                for rr, cc, hh, ww in boxes[j + 1:])
                islands += len(_components(info["board"] == carrier)) > 1
        regimes.add(len(roles) > 1)
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(frame_shapes) >= 10 and len(layouts) >= 160
    assert len(masks) >= 100 and len(palettes) >= 30
    assert counts == set(range(10)) and inlay_counts == {1, 2, 3, 4}
    assert peg_counts == {1, 2, 3, 4, 5} and regimes == {False, True}
    assert all(v >= 10 for v in endpoints.values())
    assert exterior_counts == set(range(10)) and operations == set(range(8))
    assert adjacent >= 10 and islands >= 100 and repeated >= 20 and multiside >= 30
    # Reproducibility, all legal counts, and rejection of illegal values/types.
    assert _puzzle_key(generate_puzzle(83)) == _puzzle_key(generate_puzzle(83))
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(1000 + 10 * n + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "board_shapes": len(shapes),
            "frame_shapes": len(frame_shapes), "layouts": len(layouts), "mask_orbits": len(masks),
            "palettes": len(palettes), "enclosed_counts": sorted(counts), "inlay_counts": sorted(inlay_counts),
            "peg_counts": sorted(peg_counts), "exterior_counts": sorted(exterior_counts),
            "boundary_cases": endpoints, "adjacent_pairs": adjacent, "island_pairs": islands,
            "repeated_payload_pairs": repeated, "multiside_pairs": multiside,
            "semantic_verification": "author-owned invariants and solver; no independent approval"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
