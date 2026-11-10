import numpy as np


# Exact source inputs are kept solely for the no-copy validation.
_SOURCE_INPUTS = (
    ("0000000000", "0088888880", "0088828880", "0088888880",
     "0088888880", "0088888880", "0088828880", "0088888880",
     "0000000000", "0000000000"),
    ("0000000000", "0000400000", "0004440000", "0044442000",
     "0444444400", "0044442000", "0004440000", "0000400000",
     "0000000000", "0000000000"),
    ("0000000000", "0033333000", "0003333000", "0003333000",
     "0203323000", "0333333300", "0333333300", "0333333300",
     "0003330000", "0000033000"),
    ("0000000000", "0000000000", "0577777500", "0007770000",
     "0007770000", "0007770000", "0007770000", "0777777700",
     "0000000000", "0000000000"),
    ("0000110000", "0001111000", "0011111100", "0211111100",
     "0000100000", "0000100000", "0211111100", "0011111000",
     "0001110000", "0000100000"),
    ("0000000000", "0000000000", "0992999900", "0992999900",
     "0999999900", "0000900000", "0999999900", "0999999900",
     "0999999900", "0000000000"),
)


def _counts(rng, nt, ns):
    for value, lo, hi in ((nt, 2, 10), (ns, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError("Illegal pair count")
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5)
             if a > b and (nt is None or a == nt)
             and (ns is None or b == ns)]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    return legal[int(rng.integers(len(legal)))]


def _connected(mask):
    cells = {tuple(p) for p in np.argwhere(mask)}
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in cells and p not in seen:
                seen.add(p)
                stack.append(p)
    return len(seen) == len(cells)


def _mask(kind, h, w, rng):
    y, x = np.indices((h, w))
    cy, cx = (h - 1) / 2, (w - 1) / 2
    if kind == 0:
        m = np.ones((h, w), bool)
        row = int(rng.integers(1, h - 1))
        depth = int(rng.integers(1, max(2, w // 3 + 1)))
        m[row, :depth] = False
    elif kind == 1:
        m = abs(y - cy) / (h / 2) + abs(x - cx) / (w / 2) <= 1.25
    elif kind == 2:
        half = 0.8 + abs(y - cy) / max(1, cy) * (w / 2 - 0.6)
        m = abs(x - cx) <= half
    elif kind == 3:
        m = np.zeros((h, w), bool)
        for r in range(h):
            a = int(rng.integers(0, max(1, w // 3)))
            b = int(rng.integers(0, max(1, w // 3)))
            m[r, a:w - b] = True
    elif kind == 4:
        t = y / max(1, h - 1)
        m = abs(x - cx) <= 0.7 + t * (w / 2 - 0.4)
    elif kind == 5:
        m = (abs(x - cx) <= max(1, w / 5)) | (
            abs(y - cy) <= max(1, h / 5))
    elif kind == 6:
        m = (y < 2) | (y >= h - 2) | (abs(x - cx) <= max(1, w / 5))
    elif kind == 7:
        m = (y < 2) | (y >= h - 2) | (x < 2) | (x >= w - 2)
    elif kind == 8:
        phase = float(rng.uniform(0, 2 * np.pi))
        center = cx + (w - 3) * .22 * np.sin(
            2 * np.pi * y / max(3, h - 1) + phase)
        m = abs(x - center) <= max(1.5, w / 3)
    else:
        m = (x < 2) | (x >= w - 2) | (abs(y - cy) <= 1)
    if rng.integers(2):
        m = np.fliplr(m)
    if rng.integers(2):
        m = np.flipud(m)
    if rng.random() < .3 and h != w:
        m = m.T
    return m


def _base(rng, kind, context, background, object_colors, top_mode,
          repeat=False):
    for _ in range(100):
        if repeat:
            H = int(rng.integers(10, 19))
            W = int(rng.integers(10, 19))
        elif context == "solid" and rng.random() < .27:
            H = W = 10
        else:
            lo = 12 if context == "multiple" else 8
            H, W = map(int, rng.integers(lo, 23, size=2))

        if repeat:
            h = int(rng.integers(4, min(9, H - 3)))
            w = int(rng.integers(4, min(9, W - 3)))
            r0 = int(rng.integers(2, H - h))
            c0 = int(rng.integers(2, W - w))
            local = np.ones((h, w), bool)
        else:
            hmax = min(13, H - (top_mode in ("empty", "bottom")))
            wmax = min(13, W - 1)
            if hmax < 4 or wmax < 4:
                continue
            h = H if top_mode == "both" else int(rng.integers(4, hmax + 1))
            w = int(rng.integers(4, wmax + 1))
            local = _mask(kind, h, w, rng)
            h, w = local.shape
            if h > H or w >= W:
                continue
            if top_mode in ("empty", "bottom") and h >= H:
                continue
            if top_mode == "top" or top_mode == "both":
                r0 = 0
            elif top_mode == "bottom":
                r0 = H - h
            else:
                if H - h < 1:
                    continue
                r0 = int(rng.integers(1, H - h + 1))
            c0 = int(rng.integers(1, W - w + 1))

        main = np.zeros((H, W), bool)
        main[r0:r0 + h, c0:c0 + w] = local
        if main[:, 0].any() or np.count_nonzero(main) < 14:
            continue
        if top_mode in ("top", "both") and not main[0].any():
            continue
        if top_mode in ("bottom", "both") and not main[-1].any():
            continue
        if top_mode == "empty" and main[0].any():
            continue
        rr, cc = np.where(main)
        if np.ptp(rr) < 2 or np.ptp(cc) < 2:
            continue
        obj = int(rng.choice(object_colors))
        base = np.full((H, W), background, dtype=np.int8)
        base[main] = obj

        if context == "multiple":
            sites = []
            for r in range(0 if top_mode in ("top", "both") else 1,
                           H - 1):
                for c in range(1, W - 1):
                    block = main[max(0, r - 1):min(H, r + 3),
                                 max(0, c - 1):min(W, c + 3)]
                    if not block.any():
                        sites.append((r, c))
            if not sites:
                continue
            r, c = sites[int(rng.integers(len(sites)))]
            base[r:r + 2, c:c + 2] = obj

        if context == "textured":
            colors = [c for c in object_colors if c != obj]
            if not colors:
                continue
            cells = np.argwhere(main)
            amount = min(len(cells) - 9, int(rng.integers(4, 9)))
            if amount < 4:
                continue
            chosen = rng.choice(len(cells), amount, replace=False)
            texture = int(rng.choice(colors))
            for r, c in cells[chosen]:
                base[int(r), int(c)] = texture
        return base, main, obj
    raise RuntimeError("Could not construct object context")


def _marker_options(base, main, obj, orientation, placement, spacing):
    H, W = base.shape
    inside = {tuple(map(int, p)) for p in np.argwhere(main & (base == obj))}
    outside = set()
    for r, c in np.argwhere(main):
        r, c = int(r), int(c)
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= p[0] < H and 1 <= p[1] < W and not main[p]:
                if base[p] == base[0, 0]:
                    outside.add(p)
    positions = sorted(inside | outside)
    groups = {}
    axis = 1 if orientation == "vertical" else 0
    for p in positions:
        groups.setdefault(p[axis], []).append(p)
    options = []
    for group in groups.values():
        for i, p in enumerate(group):
            for q in group[i + 1:]:
                distance = abs(p[1 - axis] - q[1 - axis])
                if (spacing == "adjacent") != (distance == 1):
                    continue
                ins = int(p in inside) + int(q in inside)
                if ((placement == "embedded" and ins == 2)
                    or (placement == "adjacent" and ins == 1)
                    or (placement == "external" and ins == 0)):
                    options.append((p, q))
    return options


def _paint(base, main, obj, marker, orientation, placement, spacing, rng):
    options = _marker_options(base, main, obj, orientation, placement, spacing)
    if not options:
        return None
    for ix in rng.permutation(len(options))[:min(40, len(options))]:
        p, q = options[int(ix)]
        inp = base.copy()
        inp[p] = marker
        inp[q] = marker
        if np.count_nonzero(inp == obj) < 8:
            continue
        objects = (inp != base[0, 0]) & (inp != marker)
        if any(not any(
            0 <= r + dr < inp.shape[0]
            and 0 <= c + dc < inp.shape[1]
            and objects[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            for r, c in (p, q)):
            continue
        values, counts = np.unique(inp, return_counts=True)
        if sum(int(v != base[0, 0] and n == 2)
               for v, n in zip(values, counts)) != 1:
            continue
        out = inp.copy()
        if orientation == "vertical":
            out[:, 0] = marker
        else:
            out[0, :] = marker
        return {"input": inp, "output": out}
    return None


def _pair(rng, plan, object_colors):
    placements = list(dict.fromkeys(
        (plan["placement"], "adjacent", "embedded", "external")))
    for attempt in range(160):
        base, main, obj = _base(
            rng, plan["kind"] if attempt < 120 else 0,
            plan["context"], plan["background"],
            object_colors, plan["top"])
        pair = _paint(base, main, obj, plan["marker"], plan["orientation"],
                      placements[min(attempt // 40, len(placements) - 1)],
                      plan["spacing"], rng)
        if pair is None:
            continue
        if (plan["orientation"] == "horizontal"
            and plan["top"] in ("top", "both")
            and not np.any(
                (pair["input"][0, 1:] != plan["background"])
                & (pair["input"][0, 1:] != plan["marker"]))):
            continue
        return pair
    raise RuntimeError("Pair generation failed")


def _contrast(rng, marker, background, object_colors):
    for _ in range(80):
        base, main, obj = _base(
            rng, 0, "solid", background, object_colors, "empty",
            repeat=True)
        a = _paint(base, main, obj, marker, "vertical", "external",
                   "adjacent", rng)
        b = _paint(base, main, obj, marker, "horizontal", "external",
                   "separated", rng)
        if a is not None and b is not None:
            return a, b
    raise RuntimeError("Contrast construction failed")


def _marker_info(inp):
    background = int(inp[0, 0])
    assert np.all(inp[:, 0] == background)
    colors, counts = np.unique(inp, return_counts=True)
    candidates = [int(v) for v, n in zip(colors, counts)
                  if v != background and n == 2]
    assert len(candidates) == 1
    marker = candidates[0]
    pts = np.argwhere(inp == marker)
    assert len(pts) == 2 and np.all(pts[:, 1] > 0)
    dr = abs(int(pts[0, 0]) - int(pts[1, 0]))
    dc = abs(int(pts[0, 1]) - int(pts[1, 1]))
    assert (dr > 0 and dc == 0) or (dc > 0 and dr == 0)
    orientation = "vertical" if dc == 0 else "horizontal"
    objects = (inp != background) & (inp != marker)
    assert np.count_nonzero(objects) >= 8
    assert np.ptp(np.where(objects)[0]) >= 2
    assert np.ptp(np.where(objects)[1]) >= 2
    for r, c in pts:
        assert any(
            0 <= int(r) + dy < inp.shape[0]
            and 0 <= int(c) + dx < inp.shape[1]
            and objects[int(r) + dy, int(c) + dx]
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    return orientation, marker, background, dr + dc


def _source_copy(inp):
    return inp.shape == (10, 10) and tuple(
        "".join(map(str, row)) for row in inp) in _SOURCE_INPUTS


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[k], q[k])
        for p, q in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)

    nmarkers = 1
    if nt >= 4 and rng.random() < .45:
        nmarkers = 2
    if nt >= 6 and rng.random() < .18:
        nmarkers = 3
    palette = list(map(int, rng.permutation(np.arange(1, 10))))
    marker_colors = palette[:nmarkers]
    novel_marker = palette[nmarkers]
    background_colors = palette[nmarkers + 1:nmarkers + 3]
    object_colors = palette[nmarkers + 3:]
    assert len(object_colors) >= 3

    train_spec = [(marker, direction) for marker in marker_colors
                  for direction in ("vertical", "horizontal")]
    for _ in range(nt - len(train_spec)):
        train_spec.append((
            marker_colors[int(rng.integers(nmarkers))],
            "vertical" if rng.integers(2) else "horizontal"))
    rng.shuffle(train_spec)
    contrast = rng.random() < .48
    contrast_marker = marker_colors[0]
    contrast_indices = []
    if contrast:
        contrast_indices = [
            next(i for i, s in enumerate(train_spec)
                 if s == (contrast_marker, direction))
            for direction in ("vertical", "horizontal")]
    bg_choices = [0]
    if rng.random() < .63:
        bg_choices.append(background_colors[0])
    if nt >= 5 and rng.random() < .20:
        bg_choices.append(background_colors[1])
    if contrast and nt == 2:
        bg_choices = [int(rng.choice(bg_choices))]

    plans = []
    spacing_order = ["adjacent", "separated"]
    rng.shuffle(spacing_order)
    placement_order = ["embedded", "adjacent", "external"]
    rng.shuffle(placement_order)
    for i, (marker, direction) in enumerate(train_spec):
        spacing = spacing_order[i] if i < 2 else (
            "adjacent" if rng.random() < .35 else "separated")
        placement = placement_order[i % 3] if i < 3 else str(
            rng.choice(placement_order))
        if i < len(bg_choices):
            background = bg_choices[i]
        else:
            background = int(rng.choice(bg_choices))
        context = "solid" if i == 0 else str(
            rng.choice(["solid", "solid", "textured", "multiple"]))
        plans.append({
            "marker": marker, "orientation": direction,
            "spacing": spacing, "placement": placement,
            "background": background, "context": context,
            "kind": int(rng.integers(10)), "top": str(rng.choice(
                ["empty", "empty", "empty", "top", "bottom", "both"]))})

    # A horizontal top overwrite is used at test time only after training
    # shows the same full-border action on a nonempty top row.
    available_horizontal = [i for i, p in enumerate(plans)
                            if p["orientation"] == "horizontal"
                            and i not in contrast_indices]
    overwrite_top = bool(available_horizontal) and rng.random() < .36
    if overwrite_top:
        plans[available_horizontal[0]]["top"] = "top"
    else:
        for p in plans:
            if p["orientation"] == "horizontal":
                p["top"] = str(rng.choice(["empty", "empty", "bottom"]))

    puzzle = {"train": [], "test": []}
    contrast_pairs = None
    if contrast:
        shared_bg = plans[contrast_indices[0]]["background"]
        contrast_pairs = _contrast(rng, contrast_marker, shared_bg,
                                   object_colors)
    for i, plan in enumerate(plans):
        if i in contrast_indices:
            pair = contrast_pairs[contrast_indices.index(i)]
        else:
            pair = _pair(rng, plan, object_colors)
        puzzle["train"].append(pair)
    rng.shuffle(puzzle["train"])

    used_backgrounds = sorted({
        int(p["input"][0, 0]) for p in puzzle["train"]})
    used_contexts = sorted({
        "textured" if len(np.unique(p["input"])) >= 4 else "solid"
        for p in puzzle["train"]})
    # An extra object is a spatial change, so it can be used in tests only
    # if the training plan included it.
    if any(p["context"] == "multiple" for i, p in enumerate(plans)
           if i not in contrast_indices):
        used_contexts.append("multiple")
    train_kinds = [p["kind"] for i, p in enumerate(plans)
                   if i not in contrast_indices]
    novel_test = rng.random() < .36
    if ns == 1:
        test_directions = [
            "vertical" if rng.integers(2) else "horizontal"]
    else:
        test_mode = str(rng.choice(
            ["vertical", "horizontal", "mixed"], p=[.25, .25, .50]))
        if test_mode == "mixed":
            test_directions = ["vertical", "horizontal"] + [
                "vertical" if rng.integers(2) else "horizontal"
                for _ in range(ns - 2)]
            rng.shuffle(test_directions)
        else:
            test_directions = [test_mode] * ns
    seen_inputs = {(p["input"].shape, p["input"].tobytes())
                   for p in puzzle["train"]}
    for j in range(ns):
        for _ in range(30):
            direction = test_directions[j]
            top_modes = (["empty", "top", "bottom", "both"]
                         if direction == "vertical" or overwrite_top
                         else ["empty", "bottom"])
            marker = (novel_marker if novel_test and j == 0
                      else int(rng.choice(marker_colors)))
            if train_kinds and rng.random() < .5:
                kind = int(rng.choice(train_kinds))
            else:
                kind = int(rng.integers(10))
            plan = {
                "marker": marker, "orientation": direction,
                "spacing": "adjacent" if rng.random() < .35 else "separated",
                "placement": str(rng.choice(
                    ["embedded", "adjacent", "external"])),
                "background": int(rng.choice(used_backgrounds)),
                "context": str(rng.choice(used_contexts)),
                "kind": kind, "top": str(rng.choice(top_modes))}
            pair = _pair(rng, plan, object_colors)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen_inputs and not _source_copy(pair["input"]):
                break
        else:
            raise RuntimeError("Distinct test pair generation failed")
        seen_inputs.add(key)
        puzzle["test"].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs = set(), set()
    train_directions = set()
    marker_directions = {}
    demonstrated_top_overwrite = False
    test_records = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _source_copy(inp)
            direction, marker, background, _ = _marker_info(inp)
            expected = inp.copy()
            if direction == "vertical":
                expected[:, 0] = marker
            else:
                expected[0, :] = marker
            assert np.array_equal(out, expected)
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.tobytes())
            assert ikey not in inputs and pkey not in pairs
            inputs.add(ikey)
            pairs.add(pkey)
            if section == "train":
                train_directions.add(direction)
                marker_directions.setdefault(marker, set()).add(direction)
                if direction == "horizontal" and np.any(
                    (inp[0, 1:] != background)
                    & (inp[0, 1:] != marker)):
                    demonstrated_top_overwrite = True
            else:
                test_records.append((inp, out, direction, marker,
                                     background))
    assert train_directions == {"vertical", "horizontal"}
    assert all(dirs == train_directions
               for dirs in marker_directions.values())
    for inp, out, direction, marker, background in test_records:
        if direction == "horizontal" and np.any(
            (inp[0, 1:] != background) & (inp[0, 1:] != marker)):
            assert demonstrated_top_overwrite
        assert np.array_equal(
            _predict_from_training(puzzle["train"], inp), out)


def _predict_from_training(train, test_input):
    learned = {}
    for pair in train:
        inp, out = pair["input"], pair["output"]
        direction, marker, _, _ = _marker_info(inp)
        if np.all(out[0, :] == marker) and np.array_equal(
            out[1:, :], inp[1:, :]):
            border = "top"
        elif np.all(out[:, 0] == marker) and np.array_equal(
            out[:, 1:], inp[:, 1:]):
            border = "left"
        else:
            raise AssertionError("Training example has no full border rule")
        if direction in learned:
            assert learned[direction] == border
        learned[direction] = border
    assert set(learned) == {"vertical", "horizontal"}
    direction, marker, _, _ = _marker_info(test_input)
    answer = test_input.copy()
    if learned[direction] == "left":
        answer[:, 0] = marker
    else:
        answer[0, :] = marker
    return answer


def _object_signature(inp):
    _, marker, background, _ = _marker_info(inp)
    mask = (inp != background) & (inp != marker)
    rr, cc = np.where(mask)
    crop = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    return crop.shape, crop.tobytes()


def self_test() -> None:
    puzzles = []
    for seed in range(1000, 1042):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        assert _same(p, generate_puzzle(seed))
        puzzles.append(p)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert not _same(puzzles[0], puzzles[1])

    sizes, shapes, palettes = set(), set(), set()
    adjacent = {"vertical": 0, "horizontal": 0}
    separated = {"vertical": 0, "horizontal": 0}
    top_vertical = top_horizontal = zero_bg = nonzero_bg = 0
    novel = familiar = single = multi = contrasts = 0
    test_orientations = set()
    multi_test_modes = set()
    background_varied = textured = disconnected = 0
    train_top_overwrite = test_top_overwrite = 0
    neighbor_styles = set()
    for p in puzzles:
        train_markers = {
            _marker_info(x["input"])[1] for x in p["train"]}
        test_markers = {
            _marker_info(x["input"])[1] for x in p["test"]}
        single += len(train_markers) == 1
        multi += len(train_markers) > 1
        novel += bool(test_markers - train_markers)
        familiar += bool(test_markers & train_markers)
        palettes.add(tuple(sorted(train_markers)))
        dirs = tuple(sorted(_marker_info(x["input"])[0]
                            for x in p["test"]))
        test_orientations.add(dirs)
        if len(dirs) >= 2:
            multi_test_modes.add(
                "mixed" if len(set(dirs)) == 2 else dirs[0])
        background_varied += len({
            _marker_info(x["input"])[2]
            for section in ("train", "test") for x in p[section]}) > 1
        signatures = [_object_signature(x["input"]) for x in p["train"]]
        contrasts += len(signatures) != len(set(signatures))
        for section in ("train", "test"):
            for pair in p[section]:
                inp, out = pair["input"], pair["output"]
                direction, marker, bg, distance = _marker_info(inp)
                assert np.array_equal(
                    _predict_from_training(p["train"], inp), out)
                (adjacent if distance == 1 else separated)[direction] += 1
                top_vertical += direction == "vertical" and np.any(
                    inp[0, 1:] != bg)
                top_horizontal += direction == "horizontal" and np.any(
                    (inp[0, 1:] != bg) & (inp[0, 1:] != marker))
                if direction == "horizontal" and np.any(
                    (inp[0, 1:] != bg) & (inp[0, 1:] != marker)):
                    if section == "train":
                        train_top_overwrite += 1
                    else:
                        test_top_overwrite += 1
                zero_bg += bg == 0
                nonzero_bg += bg != 0
                textured += len(np.unique(inp)) == 4
                objects = (inp != bg) & (inp != marker)
                disconnected += not _connected(objects)
                neighbors = []
                for r, c in np.argwhere(inp == marker):
                    neighbors.append(sum(
                        0 <= r + dr < inp.shape[0]
                        and 0 <= c + dc < inp.shape[1]
                        and objects[r + dr, c + dc]
                        for dr, dc in (
                            (-1, 0), (1, 0), (0, -1), (0, 1))))
                neighbor_styles.add(
                    "dense" if min(neighbors) >= 2 else
                    "sparse" if max(neighbors) <= 1 else "mixed")
                sizes.add(inp.shape)
                shapes.add(_object_signature(inp))
    assert all(adjacent.values()) and all(separated.values())
    assert top_vertical and top_horizontal
    assert zero_bg and nonzero_bg
    assert novel and familiar and single and multi and contrasts
    assert any(all(x == "vertical" for x in d) for d in test_orientations)
    assert any(all(x == "horizontal" for x in d) for d in test_orientations)
    assert multi_test_modes == {
        "vertical", "horizontal", "mixed"}
    assert background_varied and textured and disconnected
    assert train_top_overwrite and test_top_overwrite
    assert neighbor_styles == {"dense", "sparse", "mixed"}
    assert (10, 10) in sizes and len(sizes) >= 35
    assert len(shapes) >= 100 and len(palettes) >= 15

    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        p = generate_puzzle(568, nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
        assert _same(p, generate_puzzle(568, nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (False, 1), (2, True)):
        try:
            generate_puzzle(1, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
