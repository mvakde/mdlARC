"""Generate color-anchor stamp puzzles in the family of the staged ARC task."""

import hashlib
import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if (r, c) != (0, 0))
_SOURCE_INPUT_DIGESTS = {
    "eb437bf7df682959952d5dbf72884cb891e04d20b4834c50a9000a3983bd1b1b",
    "0b3f0b985c673696154e54270a34c0267be5a1bc0e554d2df5b2ec97a4b0c96e",
    "3b0a72812da5f18f32fa4df2ca136393d346d409542817a26efea2f130567ed3",
    "03447b9a8aa16044d13658a543710b13b20dcfeb9de1ada0f5af9d42c21314a4",
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, rng, num_train, num_test):
    a = _count("num_train", num_train, 2, 10)
    b = _count("num_test", num_test, 1, 4)
    if a is not None and b is not None:
        if a <= b:
            raise ValueError("training count must exceed test count")
        return a, b
    if a is None and b is None:
        b = 1 + seed % 4
        a = b + 1 + (seed // 4) % (10 - b)
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    else:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    return a, b


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            part = []
            todo = [(r, c)]
            seen[r, c] = True
            while todo:
                x, y = todo.pop()
                part.append((x, y))
                for dx, dy in _NEIGHBORS:
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < h and 0 <= yy < w and mask[xx, yy] and not seen[xx, yy]:
                        seen[xx, yy] = True
                        todo.append((xx, yy))
            result.append(part)
    return result


def _stamp(rng, force_long=False):
    for _ in range(100):
        long = force_long or rng.random() < 0.54
        if rng.random() < 0.25:
            length = int(rng.integers(4 if long else 3, 7 if long else 4))
            direction = ((1, 0), (0, 1), (1, 1), (1, -1))[int(rng.integers(4))]
            cells = {(i * direction[0], i * direction[1]) for i in range(length)}
        else:
            bh = int(rng.integers(2, 7 if long else 4))
            bw = int(rng.integers(2, 7 if long else 4))
            if long and max(bh, bw) < 4:
                bh = 4
            size = int(rng.integers(3, min(11, bh * bw + 1)))
            cells = {(int(rng.integers(bh)), int(rng.integers(bw)))}
            for _ in range(100):
                if len(cells) >= size:
                    break
                points = sorted(cells)
                x, y = points[int(rng.integers(len(points)))]
                dx, dy = _NEIGHBORS[int(rng.integers(8))]
                if 0 <= x + dx < bh and 0 <= y + dy < bw:
                    cells.add((x + dx, y + dy))
        rows = [x for x, _ in cells]
        cols = [y for _, y in cells]
        if len(cells) < 3 or (force_long and max(max(rows)-min(rows)+1, max(cols)-min(cols)+1) < 4):
            continue
        anchor = sorted(cells)[int(rng.integers(len(cells)))]
        return tuple(sorted((x-anchor[0], y-anchor[1]) for x, y in cells))
    raise RuntimeError("stamp sampling failed")


def _field_size(rng, n, regime):
    # Aspect ratio is independent of the stamp-source side plan.
    mode = int(rng.integers(3))
    minimum = 6 if n <= 3 else 8 if n <= 5 else 10
    h = int(rng.integers(minimum, 17))
    w = int(rng.integers(minimum, 18))
    if mode == 0:
        h, w = min(h, w), max(h, w)
    elif mode == 1:
        h, w = max(h, w), min(h, w)
    else:
        w = max(minimum, min(17, h + int(rng.integers(-2, 3))))
    return h, w


def _field_anchors(rng, h, w, stamps):
    used = set()
    anchors = []
    for stamp in stamps:
        candidates = []
        for r in range(h):
            for c in range(w):
                footprint = {(r+dr, c+dc) for dr, dc in stamp}
                if all(0 <= x < h and 0 <= y < w for x, y in footprint):
                    candidates.append((r, c, footprint))
        order = rng.permutation(len(candidates))
        if rng.random() < 0.38:
            order = sorted(order, key=lambda i: min(candidates[i][0], candidates[i][1],
                                                    h-1-candidates[i][0], w-1-candidates[i][1]))
        chosen = None
        for i in order:
            r, c, footprint = candidates[int(i)]
            if footprint & used:
                continue
            if any(max(abs(r-x), abs(c-y)) <= 1 for x, y in anchors):
                continue
            if any((x, y) in footprint for x, y in anchors):
                continue
            chosen = (r, c, footprint)
            break
        if chosen is None:
            return None
        r, c, footprint = chosen
        anchors.append((r, c))
        used.update(footprint)
    return anchors


def _side_plan(rng, n, regime):
    sides = ("top", "bottom", "left", "right")
    one_side = rng.random() < (0.39 if regime == "source_train" else 0.30)
    if n == 1 or one_side:
        return [sides[int(rng.integers(4))]] * n
    plan = [sides[int(i)] for i in rng.choice(4, size=min(n, 4), replace=False)]
    while len(plan) < n:
        plan.append(sides[int(rng.integers(4))])
    rng.shuffle(plan)
    return plan


def _margins(rng, h, w, stamps, sides):
    required = {side: 1 for side in ("top", "bottom", "left", "right")}
    for stamp, side in zip(stamps, sides):
        rows = [r for r, _ in stamp]
        cols = [c for _, c in stamp]
        extent = max(rows)-min(rows)+1 if side in ("top", "bottom") else max(cols)-min(cols)+1
        required[side] = max(required[side], extent+1)
    margins = {}
    for side in required:
        if required[side] == 1:
            margins[side] = int(rng.choice((1, 1, 2, 3, 4, 5, 7)))
        else:
            margins[side] = required[side] + int(rng.integers(0, 4))
    if h + margins["top"] + margins["bottom"] > 30:
        return None
    if w + margins["left"] + margins["right"] > 30:
        return None
    return margins


def _external_anchor(rng, stamp, side, shape, box, occupied):
    gh, gw = shape
    r0, c0, r1, c1 = box
    candidates = []
    for r in range(gh):
        for c in range(gw):
            pos = {(r+dr, c+dc) for dr, dc in stamp}
            if not all(0 <= x < gh and 0 <= y < gw for x, y in pos):
                continue
            if (0, 0) in pos or (0, gw-1) in pos or (gh-1, 0) in pos or (gh-1, gw-1) in pos:
                continue
            xs = [x for x, _ in pos]
            ys = [y for _, y in pos]
            if side == "top" and (max(xs) > r0-2 or max(ys) < c0 or min(ys) > c1):
                continue
            if side == "bottom" and (min(xs) < r1+2 or max(ys) < c0 or min(ys) > c1):
                continue
            if side == "left" and (max(ys) > c0-2 or max(xs) < r0 or min(xs) > r1):
                continue
            if side == "right" and (min(ys) < c1+2 or max(xs) < r0 or min(xs) > r1):
                continue
            if any((x+dx, y+dy) in occupied for x, y in pos for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            candidates.append((r, c, pos))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _make_pair(rng, regime, index, repeat=False, carrier_preference=None):
    for _ in range(300):
        if regime == "source_train" and index == 0:
            n = int(rng.integers(2, 5))
        elif regime == "source_train" and index == 2 and rng.random() < 0.6:
            n = 1
        else:
            choices = (1, 2, 2, 3, 3, 4, 5, 6) if regime == "source_train" else (1, 2, 3, 4, 4, 5, 6, 7)
            n = int(choices[int(rng.integers(len(choices)))])
        max_d = min(3, 7-n)
        if regime == "source_train" and index == 0:
            d = int(rng.integers(1, max_d+1))
        else:
            d = int(rng.integers(max_d+1))
        h, w = _field_size(rng, n, regime)
        palette = [int(x) for x in rng.permutation(10)]
        if carrier_preference is not None:
            position = palette.index(carrier_preference)
            palette[1], palette[position] = palette[position], palette[1]
        background, carrier, drawing = palette[:3]
        identities = palette[3:3+n+d]
        stamps = [_stamp(rng, force_long=(regime == "source_train" and index == 0 and i == 0))
                  for i in range(n+d)]
        if n >= 2 and rng.random() < 0.22:
            stamps[1] = stamps[0]  # duplicate offset masks are legal under distinct identities
        repeat_index = int(rng.integers(n)) if repeat else None
        field_stamps = stamps[:n] + ([stamps[repeat_index]] if repeat else [])
        field_identities = identities[:n] + ([identities[repeat_index]] if repeat else [])
        anchors = _field_anchors(rng, h, w, field_stamps)
        if anchors is None:
            continue
        sides = _side_plan(rng, n, regime)
        sides += [("top", "bottom", "left", "right")[int(rng.integers(4))] for _ in range(d)]
        margins = _margins(rng, h, w, stamps, sides)
        if margins is None:
            continue
        gh = margins["top"] + h + margins["bottom"]
        gw = margins["left"] + w + margins["right"]
        r0, c0 = margins["top"], margins["left"]
        box = (r0, c0, r0+h-1, c0+w-1)
        field = np.full((h, w), carrier, dtype=np.int8)
        output = field.copy()
        for identity, stamp, (r, c) in zip(field_identities, field_stamps, anchors):
            field[r, c] = identity
            for dr, dc in stamp:
                output[r+dr, c+dc] = identity if (dr, dc) == (0, 0) else drawing
        # All anchor cells have been set before output reconstruction is finalized.
        output[field != carrier] = field[field != carrier]
        grid = np.full((gh, gw), background, dtype=np.int8)
        grid[r0:r0+h, c0:c0+w] = field
        occupied = {(r, c) for r in range(r0, r0+h) for c in range(c0, c0+w)}
        success = True
        for identity, stamp, side in zip(identities, stamps, sides):
            placed = _external_anchor(rng, stamp, side, (gh, gw), box, occupied)
            if placed is None:
                success = False
                break
            ar, ac, pos = placed
            for dr, dc in stamp:
                grid[ar+dr, ac+dc] = identity if (dr, dc) == (0, 0) else drawing
            occupied.update(pos)
        if success:
            return grid, output
    raise RuntimeError(f"could not place a {regime} pair")


def _analyze(inp, out):
    gh, gw = inp.shape
    bg = int(inp[0, 0])
    assert all(int(inp[r, c]) == bg for r, c in ((0, gw-1), (gh-1, 0), (gh-1, gw-1)))
    components = _components(inp != bg)
    assert len(components) >= 2
    sizes = [len(x) for x in components]
    field_index = int(np.argmax(sizes))
    assert sizes.count(sizes[field_index]) == 1
    part = components[field_index]
    r0, r1 = min(r for r, _ in part), max(r for r, _ in part)
    c0, c1 = min(c for _, c in part), max(c for _, c in part)
    h, w = r1-r0+1, c1-c0+1
    assert len(part) == h*w and 5 <= h <= 20 and 5 <= w <= 20
    patch = inp[r0:r1+1, c0:c1+1]
    assert out.shape == (h, w) and np.all(patch != bg) and inp.shape != out.shape
    outside = [x for i, x in enumerate(components) if i != field_index]
    assert sizes[field_index] > 3*max(map(len, outside))
    colors, counts = np.unique(patch, return_counts=True)
    carrier = int(colors[np.argmax(counts)])
    assert int(np.max(counts)) > h*w//2
    markers = [(r, c, int(patch[r, c])) for r in range(h) for c in range(w)
               if int(patch[r, c]) != carrier]
    assert 1 <= len(markers) <= 8
    marker_colors = {v for _, _, v in markers}
    assert bg not in marker_colors
    for i, (r, c, _) in enumerate(markers):
        assert all(max(abs(r-x), abs(c-y)) > 1 for x, y, _ in markers[:i])
    common = None
    for part in outside:
        col = {int(inp[r, c]) for r, c in part}
        assert len(col) == 2 and 3 <= len(part) <= 12
        common = col if common is None else common & col
    possible = common - marker_colors - {bg, carrier}
    assert len(possible) == 1
    drawing = possible.pop()
    assert drawing not in marker_colors
    sources = {}
    stamp_masks = []
    for part in outside:
        identity_cells = [(r, c) for r, c in part if int(inp[r, c]) != drawing]
        assert len(identity_cells) == 1
        ar, ac = identity_cells[0]
        identity = int(inp[ar, ac])
        assert identity not in sources and identity not in (bg, carrier)
        offsets = tuple(sorted((r-ar, c-ac) for r, c in part))
        assert max(r for r, _ in part)-min(r for r, _ in part)+1 <= 6
        assert max(c for _, c in part)-min(c for _, c in part)+1 <= 6
        sources[identity] = offsets
        stamp_masks.append(offsets)
    assert marker_colors <= set(sources)
    rebuilt = patch.copy()
    occupied = set()
    for r, c, identity in markers:
        for dr, dc in sources[identity]:
            x, y = r+dr, c+dc
            assert 0 <= x < h and 0 <= y < w and (x, y) not in occupied
            assert int(patch[x, y]) == (identity if (dr, dc) == (0, 0) else carrier)
            rebuilt[x, y] = identity if (dr, dc) == (0, 0) else drawing
            occupied.add((x, y))
    assert np.array_equal(rebuilt, out)
    return {"background": bg, "carrier": carrier, "drawing": drawing,
            "field_bbox": (r0, c0, r1, c1), "field_shape": (h, w),
            "margins": (r0, gh-1-r1, c0, gw-1-c1),
            "markers": markers, "marker_count": len(markers),
            "source_count": len(sources), "distractor_count": len(set(sources)-marker_colors),
            "masks": stamp_masks, "matched_masks": [sources[v] for _, _, v in markers],
            "sides": [_side_of(inp, drawing, v, (r0, c0, r1, c1)) for _, _, v in markers]}


def _side_of(inp, drawing, identity, box):
    for part in _components((inp == drawing) | (inp == identity)):
        if sum(int(inp[r, c]) == identity for r, c in part) != 1:
            continue
        rows = [r for r, _ in part]
        cols = [c for _, c in part]
        r0, c0, r1, c1 = box
        if max(rows) < r0:
            return "top"
        if min(rows) > r1:
            return "bottom"
        if max(cols) < c0:
            return "left"
        if min(cols) > c1:
            return "right"
    raise AssertionError("source side missing")


def _grid_key(a):
    return (a.shape, a.astype(np.uint8).tobytes())


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    used = set()
    repeat_enabled = rng.random() < 0.55
    training_markers = set()
    training_carriers = set()
    for regime, amount, key in (("source_train", train_count, "train"),
                                ("target_test", test_count, "test")):
        for index in range(amount):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
            repeat = (repeat_enabled and index == 1 if key == "train" else
                      repeat_enabled and rng.random() < 0.28)
            carrier_preference = None
            if key == "test":
                unseen = sorted(set(range(10)) - training_markers)
                choice = rng.random()
                if choice < 0.38 and unseen:
                    carrier_preference = unseen[int(rng.integers(len(unseen)))]
                elif choice < 0.70:
                    familiar = sorted(training_carriers)
                    carrier_preference = familiar[int(rng.integers(len(familiar)))]
            for _ in range(20):
                inp, out = _make_pair(pair_rng, regime, index, repeat,
                                      carrier_preference)
                token = _grid_key(inp)
                if token not in used:
                    used.add(token)
                    puzzle[key].append({"input": inp, "output": out})
                    if key == "train":
                        info = _analyze(inp, out)
                        training_carriers.add(info["carrier"])
                        training_markers.update(v for _, _, v in info["markers"])
                    break
            else:
                raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            token = _grid_key(inp)
            assert token not in seen
            seen.add(token)
            digest = hashlib.sha256(bytes(inp.shape) + inp.astype(np.uint8).tobytes()).hexdigest()
            assert digest not in _SOURCE_INPUT_DIGESTS
            info = _analyze(inp, out)
            infos[split].append(info)
    # Distractor rejection is demonstrated in training, even at the 2/1 boundary.
    assert any(x["distractor_count"] for x in infos["train"])
    assert any(any(max(max(r for r, _ in mask)-min(r for r, _ in mask)+1,
                       max(c for _, c in mask)-min(c for _, c in mask)+1) >= 4
                   for mask in x["matched_masks"]) for x in infos["train"])
    if any(len({v for _, _, v in x["markers"]}) < x["marker_count"]
           for x in infos["test"]):
        assert any(len({v for _, _, v in x["markers"]}) < x["marker_count"]
                   for x in infos["train"])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    stats = {"long": 0, "edge_marker": 0, "narrow_margin": 0,
             "train_dispersed": 0, "test_one_side": 0, "test_portrait_square": 0,
             "multiple_distractors": 0, "single_match": 0, "duplicate_mask": 0,
             "vary_background": 0, "vary_drawing": 0, "repeat_marker": 0,
             "carrier_majority": 0, "corner_identity": 0, "endpoint_identity": 0,
             "mask_size_variety": 0, "very_long": 0, "long_and_edge": 0,
             "test_carrier_variety": 0, "familiar_test_carrier": 0,
             "reassigned_test_carrier": 0, "familiar_test_size": 0}
    shapes = set()
    layouts = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        data = {s: [_analyze(x["input"], x["output"]) for x in puzzle[s]]
                for s in ("train", "test")}
        all_info = data["train"] + data["test"]
        stats["vary_background"] += len({x["background"] for x in all_info}) > 1
        stats["vary_drawing"] += len({x["drawing"] for x in all_info}) > 1
        train_carriers = {x["carrier"] for x in data["train"]}
        train_markers = {v for x in data["train"] for _, _, v in x["markers"]}
        stats["test_carrier_variety"] += len({x["carrier"] for x in data["test"]}) > 1
        stats["familiar_test_carrier"] += any(x["carrier"] in train_carriers
                                              for x in data["test"])
        stats["reassigned_test_carrier"] += any(x["carrier"] in train_markers
                                                for x in data["test"])
        stats["familiar_test_size"] += any(
            x["field_shape"][0] <= max(y["field_shape"][0] for y in data["train"])
            and x["field_shape"][1] <= max(y["field_shape"][1] for y in data["train"])
            for x in data["test"])
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], data[split]):
                h, w = info["field_shape"]
                shapes.add((h, w))
                layouts.add((info["field_bbox"], tuple((r, c) for r, c, _ in info["markers"]),
                             tuple(info["sides"])))
                stats["narrow_margin"] += min(info["margins"]) == 1
                stats["multiple_distractors"] += info["distractor_count"] >= 2
                stats["single_match"] += info["marker_count"] == 1
                stats["duplicate_mask"] += len(set(info["masks"])) < len(info["masks"])
                stats["repeat_marker"] += len({v for _, _, v in info["markers"]}) < info["marker_count"]
                stats["carrier_majority"] += int(np.sum(pair["input"] == info["carrier"])) > int(
                    np.sum(pair["input"] == info["background"]))
                stats["mask_size_variety"] += len({len(mask) for mask in info["masks"]}) > 1
                for mask in info["masks"]:
                    rs = [r for r, _ in mask]
                    cs = [c for _, c in mask]
                    stats["corner_identity"] += (0 in (min(rs), max(rs))
                                                 and 0 in (min(cs), max(cs)))
                    stats["endpoint_identity"] += sum(
                        max(abs(r), abs(c)) == 1 for r, c in mask
                        if (r, c) != (0, 0)) == 1
                stats["edge_marker"] += any(r in (0, h-1) or c in (0, w-1)
                                            for r, c, _ in info["markers"])
                has_long = any(max(max(r for r, _ in mask)-min(r for r, _ in mask)+1,
                                   max(c for _, c in mask)-min(c for _, c in mask)+1) >= 4
                               for mask in info["matched_masks"])
                stats["long"] += has_long
                stats["very_long"] += any(max(max(r for r, _ in mask)-min(r for r, _ in mask)+1,
                                               max(c for _, c in mask)-min(c for _, c in mask)+1) >= 5
                                          for mask in info["matched_masks"])
                stats["long_and_edge"] += has_long and any(
                    r in (0, h-1) or c in (0, w-1) for r, c, _ in info["markers"])
                if split == "train":
                    stats["train_dispersed"] += len(set(info["sides"])) >= 2
                else:
                    stats["test_one_side"] += len(set(info["sides"])) == 1
                    stats["test_portrait_square"] += h >= w
    assert len(shapes) >= 35 and len(layouts) >= 120
    assert all(value > 0 for value in stats.values()), stats
    assert _equal(generate_puzzle(12345, 6, 2), generate_puzzle(12345, 6, 2))
    assert not _equal(generate_puzzle(700), generate_puzzle(701))
    for seed, a, b in ((2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
                       (3001, None, 1), (3002, None, 4), (3003, 2, None),
                       (3004, 10, None)):
        p = generate_puzzle(seed, a, b)
        assert a is None or len(p["train"]) == a
        assert b is None or len(p["test"]) == b
        validate_generated_puzzle(p)
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4), (True, 1), (3, False)):
        try:
            generate_puzzle(4001, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
