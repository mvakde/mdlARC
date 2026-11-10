import numpy as np


_ORIGINAL_INPUTS = frozenset(
    tuple(block.splitlines())
    for block in (
        """444444444444444444
444644444444644444
441114444444444444
444144444444444444
444444444444444444
444444444444544444
444444444444444444
444544444444444444
444444444444444444
444444444444444444
444444111444444444
444444141444444444
444344434444443444
444444444444444444
444444444444444444
444444454444444444
444444444444444444
444444444444444444""",
        """444444444444444444
411144444444444444
412144424444424444
411144444444444444
444444444444444444
444444444444544444
444444444444444444
441444444444444444
411144444444444444
413144444344444444
414144444444444444
444444444444444444
444444544444444444
444444444444444444
444444444444411144
444844444444148414
444444444444414144
444444444444444444""",
        """444444444444444444
444444444444444444
444414444444444444
444414444444444444
444161444644644644
444414444444444444
444444445444444444
444444444444444444
444144444444444444
441414444444444444
444344444434444444
444444444444444454
444444444444444444
444444444424244444
444444444422244444
484448444428244444
444444444444444444
444444444444444444""",
    )
)


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} is outside its legal range")
    choices = [
        (a, b)
        for a in range(2, 11)
        for b in range(1, 5)
        if a > b
        and (num_train is None or a == int(num_train))
        and (num_test is None or b == int(num_test))
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _normalize(points):
    low_y = min(y for y, _ in points)
    low_x = min(x for _, x in points)
    return tuple(sorted((y - low_y, x - low_x) for y, x in points))


def _sample_shape(rng):
    """Compose an eight-connected template; four-connectivity is optional."""
    mode = int(rng.choice(6, p=[0.17, 0.16, 0.20, 0.17, 0.19, 0.11]))
    if mode == 0:  # filled rectangles, including the source's 3 by 3 case
        h, w = [(2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)][
            int(rng.integers(6))
        ]
        points = {(y, x) for y in range(h) for x in range(w)}
    elif mode == 1:  # outlines, with genuine background holes
        h = int(rng.integers(3, 5))
        w = int(rng.integers(3, 6))
        points = {
            (y, x)
            for y in range(h)
            for x in range(w)
            if y in (0, h - 1) or x in (0, w - 1)
        }
        if rng.random() < 0.35:
            points.remove((int(rng.choice([0, h - 1])), int(rng.integers(1, w - 1))))
    elif mode == 2:  # diagonal growth, often entirely split under four-neighbor grouping
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        points = {(int(rng.integers(h)), int(rng.integers(w)))}
        target = int(rng.integers(4, min(9, (h * w + 1) // 2) + 1))
        while len(points) < target:
            candidates = sorted({
                (y + dy, x + dx)
                for y, x in points
                for dy in (-1, 1)
                for dx in (-1, 1)
                if 0 <= y + dy < h and 0 <= x + dx < w
                and (y + dy, x + dx) not in points
            })
            if not candidates:
                break
            points.add(candidates[int(rng.integers(len(candidates)))])
        if len(points) < 4 or len({y for y, _ in points}) == 1 or len({x for _, x in points}) == 1:
            return _sample_shape(rng)
    elif mode == 3:  # T, cross, and offset junctions
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        bar_y = int(rng.choice([0, h // 2, h - 1]))
        stem_x = int(rng.integers(1, w - 1))
        points = {(bar_y, x) for x in range(w)} | {(y, stem_x) for y in range(h)}
        if rng.random() < 0.40:
            points.add((int(rng.integers(h)), max(0, stem_x - 1)))
    elif mode == 4:  # unconstrained irregular growth in eight-neighbor space
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        points = {(int(rng.integers(h)), int(rng.integers(w)))}
        target = int(rng.integers(4, min(13, h * w) + 1))
        while len(points) < target:
            candidates = sorted({
                (y + dy, x + dx)
                for y, x in points
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w
                and (y + dy, x + dx) not in points
            })
            if not candidates:
                break
            points.add(candidates[int(rng.integers(len(candidates)))])
    else:  # elongated narrow templates
        length = int(rng.integers(4, 7))
        points = {(1, x) for x in range(length)}
        for x in range(length):
            if rng.random() < 0.55:
                points.add((int(rng.choice([0, 2])), x))
        if rng.random() < 0.5:
            points = {(x, y) for y, x in points}
    if len({y for y, _ in points}) == 1 or len({x for _, x in points}) == 1:
        return _sample_shape(rng)
    points = _normalize(points)
    diagonal_tips = [
        p for p in points
        if all((p[0] + dy, p[1] + dx) not in points for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    ]
    if diagonal_tips and rng.random() < 0.50:
        anchor = diagonal_tips[int(rng.integers(len(diagonal_tips)))]
    elif mode == 0 and rng.random() < 0.70:
        cy = (max(y for y, _ in points)) / 2
        cx = (max(x for _, x in points)) / 2
        near = sorted(points, key=lambda p: ((p[0] - cy) ** 2 + (p[1] - cx) ** 2, p))
        anchor = near[0]
    else:
        anchor = points[int(rng.integers(len(points)))]
    return {
        "points": points,
        "anchor": anchor,
        "height": max(y for y, _ in points) + 1,
        "width": max(x for _, x in points) + 1,
    }


def _overlap(a, b, gap=0):
    ay, ax, ah, aw = a
    by, bx, bh, bw = b
    return not (
        ay + ah + gap <= by or by + bh + gap <= ay
        or ax + aw + gap <= bx or bx + bw + gap <= ax
    )


def _near(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _choose_pair_colors(rng, color_order, previous_anchors, source_count, mixed, preferred_background=None):
    background = (
        int(preferred_background) if preferred_background is not None
        else int(color_order[int(rng.integers(10))])
    )
    available = [int(c) for c in color_order if c != background]
    carrier_count = 2 if mixed and source_count >= 2 else 1
    carriers = []
    reusable = [c for c in previous_anchors if c in available]
    if reusable and rng.random() < 0.48:
        carriers.append(int(reusable[int(rng.integers(len(reusable)))]))
    remaining = [c for c in available if c not in carriers]
    rng.shuffle(remaining)
    carriers.extend(remaining[:carrier_count - len(carriers)])
    anchor_pool = [c for c in available if c not in carriers]
    rng.shuffle(anchor_pool)
    anchors = anchor_pool[:source_count]
    extras = [c for c in anchor_pool[source_count:]]
    rng.shuffle(extras)
    return background, carriers, anchors, extras


def _sample_counts(rng, source_count, maximum=4, baseline=False):
    if baseline:
        return [2, 1]
    counts = [int(rng.choice([1, 2, 3, 4], p=[0.43, 0.34, 0.18, 0.05])) for _ in range(source_count)]
    counts = [min(c, maximum) for c in counts]
    while sum(counts) > 8:
        candidates = [i for i, c in enumerate(counts) if c > 1]
        counts[candidates[int(rng.integers(len(candidates)))]] -= 1
    return counts


def _sample_size(rng, required_area, compact=False):
    for _ in range(100):
        low, high = (17, 23) if compact else (12, 31)
        h = int(rng.integers(low, high))
        w = int(rng.integers(low, high))
        if h * w >= required_area * 1.7 + 35:
            return h, w
    return (23, 23) if compact else (30, 30)


def _position_for_edge(rng, height, width, mh, mw, corner=False):
    side = int(rng.integers(4))
    if side == 0:
        top, left = 0, int(rng.integers(width - mw + 1))
    elif side == 1:
        top, left = height - mh, int(rng.integers(width - mw + 1))
    elif side == 2:
        top, left = int(rng.integers(height - mh + 1)), 0
    else:
        top, left = int(rng.integers(height - mh + 1)), width - mw
    if corner:
        if side < 2:
            left = 0 if rng.random() < 0.5 else width - mw
        else:
            top = 0 if rng.random() < 0.5 else height - mh
    return top, left


def _place_destinations(rng, motifs, counts, height, width, want_touch, want_edge, want_corner):
    boxes = []
    markers = []
    placements = []

    def add(index, top, left):
        motif = motifs[index]
        rect = (top, left, motif["height"], motif["width"])
        marker = (top + motif["anchor"][0], left + motif["anchor"][1])
        if not (0 <= top and 0 <= left and top + rect[2] <= height and left + rect[3] <= width):
            return False
        if any(_overlap(rect, old) for old in boxes) or any(_near(marker, old) for old in markers):
            return False
        boxes.append(rect)
        markers.append(marker)
        placements.append((index, top, left))
        return True

    remaining = list(counts)
    if want_touch:
        choices = [i for i, count in enumerate(counts) if count >= 2]
        if not choices:
            return None
        index = choices[int(rng.integers(len(choices)))]
        motif = motifs[index]
        for _ in range(120):
            vertical = bool(rng.integers(2))
            mh, mw = motif["height"], motif["width"]
            if vertical and 2 * mh > height or not vertical and 2 * mw > width:
                continue
            if want_edge:
                top, left = _position_for_edge(rng, height, width, mh, mw, want_corner)
            else:
                top = int(rng.integers(height - mh + 1))
                left = int(rng.integers(width - mw + 1))
            sign = -1 if rng.random() < 0.5 else 1
            second = (top + sign * mh, left) if vertical else (top, left + sign * mw)
            first_rect = (top, left, mh, mw)
            second_rect = (second[0], second[1], mh, mw)
            if not (0 <= second[0] and 0 <= second[1] and second[0] + mh <= height and second[1] + mw <= width):
                continue
            if want_edge and not any(
                r[0] == 0 or r[1] == 0 or r[0] + r[2] == height or r[1] + r[3] == width
                for r in (first_rect, second_rect)
            ):
                continue
            if want_corner and not any(
                r[0] in (0, height - r[2]) and r[1] in (0, width - r[3])
                for r in (first_rect, second_rect)
            ):
                continue
            if add(index, top, left) and add(index, *second):
                remaining[index] -= 2
                break
            boxes.clear(); markers.clear(); placements.clear()
        else:
            return None
    elif want_edge:
        index = int(rng.integers(len(motifs)))
        motif = motifs[index]
        top, left = _position_for_edge(rng, height, width, motif["height"], motif["width"], want_corner)
        if not add(index, top, left):
            return None
        remaining[index] -= 1

    pending = [i for i, count in enumerate(remaining) for _ in range(count)]
    rng.shuffle(pending)
    for index in pending:
        motif = motifs[index]
        for _ in range(350):
            top = int(rng.integers(height - motif["height"] + 1))
            left = int(rng.integers(width - motif["width"] + 1))
            if add(index, top, left):
                break
        else:
            return None
    return boxes, markers, placements


def _generate_pair(rng, color_order, previous_anchors, regime, options):
    source_count = options["source_count"]
    counts = options["counts"]
    mixed = options["mixed"]
    background, carriers, anchors, extras = _choose_pair_colors(
        rng, color_order, previous_anchors, source_count, mixed,
        options.get("preferred_background"),
    )
    carrier_assignment = [carriers[0]] * source_count
    if len(carriers) > 1:
        carrier_assignment[int(rng.integers(source_count))] = carriers[1]
    for _ in range(90):
        motifs = []
        for anchor_color, carrier_color in zip(anchors, carrier_assignment):
            motif = _sample_shape(rng)
            motif["anchor_color"] = anchor_color
            motif["carrier_color"] = carrier_color
            motifs.append(motif)
        required_area = sum(m["height"] * m["width"] * (c + 1) for m, c in zip(motifs, counts))
        height, width = _sample_size(rng, required_area, options.get("compact", False))
        arranged = _place_destinations(
            rng, motifs, counts, height, width,
            options["touch"], options["edge"], options["corner"],
        )
        if arranged is None:
            continue
        boxes, markers, placements = arranged
        sources = []
        for motif in motifs:
            for _ in range(350):
                top = int(rng.integers(height - motif["height"] + 1))
                left = int(rng.integers(width - motif["width"] + 1))
                rect = (top, left, motif["height"], motif["width"])
                if any(_overlap(rect, old, gap=1) for old in sources):
                    continue
                if any(_overlap(rect, (y, x, 1, 1), gap=1) for y, x in markers):
                    continue
                sources.append(rect)
                break
            else:
                break
        if len(sources) != source_count:
            continue
        inp = np.full((height, width), background, dtype=np.int8)
        out = np.full((height, width), background, dtype=np.int8)
        for motif, rect in zip(motifs, sources):
            for dy, dx in motif["points"]:
                inp[rect[0] + dy, rect[1] + dx] = (
                    motif["anchor_color"] if (dy, dx) == motif["anchor"] else motif["carrier_color"]
                )
        for marker, (index, _, _) in zip(markers, placements):
            inp[marker] = motifs[index]["anchor_color"]
        for index, top, left in placements:
            motif = motifs[index]
            for dy, dx in motif["points"]:
                out[top + dy, left + dx] = (
                    motif["anchor_color"] if (dy, dx) == motif["anchor"] else motif["carrier_color"]
                )
        distractor_total = options["distractors"]
        distractor_colors = extras[:min(2, distractor_total)]
        if distractor_total and not distractor_colors:
            continue
        occupied = [tuple(map(int, p)) for p in np.argwhere(inp != background)]
        # The first training example always shows several unmatched markers of one color.
        for k in range(distractor_total):
            color = distractor_colors[0] if k < 2 else distractor_colors[k % len(distractor_colors)]
            for _ in range(400):
                pos = (int(rng.integers(height)), int(rng.integers(width)))
                if any(_near(pos, old) for old in occupied):
                    continue
                if any(_overlap(rect, (pos[0], pos[1], 1, 1), gap=1) for rect in sources):
                    continue
                occupied.append(pos)
                inp[pos] = color
                break
            else:
                break
        else:
            return {"input": inp, "output": out}
    raise RuntimeError(f"Could not place {regime} pair")


def _pair_options(rng, baseline=False, max_multiplicity=4, regime="source_train", compact=False):
    if baseline:
        return {
            "source_count": 2, "counts": [2, 1], "mixed": False,
            "edge": bool(rng.random() < (0.15 if compact else 0.3)), "corner": False,
            "touch": bool(rng.random() < (0.15 if compact else 0.3)),
            "distractors": int(rng.integers(2, 4)),
        }
    source_count = int(rng.choice(
        [1, 2, 3, 4],
        p=[0.13, 0.62, 0.25, 0.0] if compact else [0.12, 0.38, 0.36, 0.14],
    ))
    counts = _sample_counts(rng, source_count, min(max_multiplicity, 3) if compact else max_multiplicity)
    touch = bool(rng.random() < (0.16 if compact else (0.38 if regime == "source_train" else 0.46)))
    if touch and max(counts) < 2:
        counts[int(rng.integers(source_count))] = 2
    edge = bool(rng.random() < (0.24 if compact else (0.43 if regime == "source_train" else 0.55)))
    corner = edge and bool(rng.random() < 0.24)
    return {
        "source_count": source_count, "counts": counts,
        "mixed": source_count >= 2 and bool(rng.random() < (0.20 if compact else 0.48)),
        "edge": edge, "corner": corner, "touch": touch,
        "distractors": int(rng.choice(
            [0, 1, 2, 3, 4],
            p=[0.17, 0.38, 0.33, 0.12, 0.0] if compact else [0.14, 0.20, 0.31, 0.23, 0.12],
        )),
    }


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Only the transformation is a learned puzzle-wide convention. This shuffled
    # palette is a sampling preference, not a fixed map of semantic color roles.
    color_order = tuple(int(v) for v in rng.permutation(10))
    compact = bool(rng.random() < 0.20)
    preferred_background = color_order[0] if compact else None
    for _ in range(24):
        pair_seeds = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.int64)
        train, test = [], []
        previous_anchors = []
        for index in range(train_count):
            pair_rng = np.random.default_rng(int(pair_seeds[index]))
            options = _pair_options(pair_rng, baseline=index == 0, compact=compact)
            options["compact"] = compact
            options["preferred_background"] = preferred_background
            pair = _generate_pair(pair_rng, color_order, previous_anchors, "source_train", options)
            train.append(pair)
            info = _analyze_pair(pair["input"], pair["output"])
            previous_anchors.extend(sorted(info["anchors"]))
        train_max = max(max(_analyze_pair(p["input"], p["output"])["counts"].values()) for p in train)
        train_mixed = any(len(_analyze_pair(p["input"], p["output"])["carriers"]) > 1 for p in train)
        for index in range(test_count):
            pair_rng = np.random.default_rng(int(pair_seeds[train_count + index]))
            options = _pair_options(
                pair_rng, max_multiplicity=train_max, regime="target_test", compact=compact
            )
            options["compact"] = compact
            options["preferred_background"] = preferred_background
            if not train_mixed:
                options["mixed"] = False
            pair = _generate_pair(pair_rng, color_order, previous_anchors, "target_test", options)
            test.append(pair)
            previous_anchors.extend(sorted(_analyze_pair(pair["input"], pair["output"])["anchors"]))
        puzzle = {"train": train, "test": test}
        all_pairs = train + test
        backgrounds = {_analyze_pair(p["input"], p["output"])["background"] for p in all_pairs}
        shapes = {p["input"].shape for p in all_pairs}
        if compact or (len(backgrounds) >= 2 and len(shapes) >= 2):
            return puzzle
    raise RuntimeError("Could not create within-puzzle variety")


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def _components(grid, background):
    height, width = grid.shape
    remaining = {(int(y), int(x)) for y, x in np.argwhere(grid != background)}
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x, int(grid[y, x])))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    point = (y + dy, x + dx)
                    if (dy or dx) and 0 <= point[0] < height and 0 <= point[1] < width and point in remaining:
                        remaining.remove(point)
                        stack.append(point)
        components.append(cells)
    return components


def _four_connected(cells):
    positions = {(y, x) for y, x, _ in cells}
    seen = {next(iter(positions))}
    stack = list(seen)
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            point = (y + dy, x + dx)
            if point in positions and point not in seen:
                seen.add(point)
                stack.append(point)
    return seen == positions


def _analyze_pair(inp, out):
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    background = int(np.bincount(inp.ravel().astype(np.int64), minlength=10).argmax())
    assert np.count_nonzero(inp == background) > inp.size // 2
    components = _components(inp, background)
    markers = {}
    sources = []
    for component in components:
        if len(component) == 1:
            y, x, color = component[0]
            markers.setdefault(color, []).append((y, x))
        else:
            assert 4 <= len(component) <= 20
            sources.append(component)
    assert 1 <= len(sources) <= 4
    assert len(markers) >= 1
    template_by_anchor = {}
    four_split = 0
    diagonal_anchors = 0
    filled = 0
    filled_center = 0
    holes = 0
    interior_gaps = 0
    enclosed_holes = 0
    branches = 0
    large = 0
    elongated = 0
    for component in sources:
        colors = {color for _, _, color in component}
        candidates = colors.intersection(markers)
        assert len(colors) == 2 and len(candidates) == 1
        anchor_color = next(iter(candidates))
        carrier = next(iter(colors - {anchor_color}))
        anchor_cells = [(y, x) for y, x, c in component if c == anchor_color]
        assert len(anchor_cells) == 1 and anchor_color not in template_by_anchor
        assert len(component) - 1 >= 3
        ay, ax = anchor_cells[0]
        positions = {(y, x) for y, x, _ in component}
        if not _four_connected(component):
            four_split += 1
        if all((ay + dy, ax + dx) not in positions for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            diagonal_anchors += 1
        y0, y1 = min(y for y, _, _ in component), max(y for y, _, _ in component)
        x0, x1 = min(x for _, x, _ in component), max(x for _, x, _ in component)
        h, w = y1 - y0 + 1, x1 - x0 + 1
        assert h >= 2 and w >= 2 and h <= 6 and w <= 6
        is_filled = len(component) == h * w
        filled += is_filled
        filled_center += is_filled and abs(ay - (y0 + y1) / 2) <= 0.5 and abs(ax - (x0 + x1) / 2) <= 0.5
        holes += len(component) < h * w
        interior_gaps += any(
            (y, x) not in positions
            for y in range(y0 + 1, y1)
            for x in range(x0 + 1, x1)
        )
        empty = {
            (y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
            if (y, x) not in positions
        }
        while empty:
            start = next(iter(empty))
            empty.remove(start)
            region = {start}
            pending = [start]
            while pending:
                y, x = pending.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (y + dy, x + dx)
                    if neighbor in empty:
                        empty.remove(neighbor)
                        region.add(neighbor)
                        pending.append(neighbor)
            enclosed_holes += all(
                y not in (y0, y1) and x not in (x0, x1) for y, x in region
            )
        branches += not is_filled and any(
            sum((y + dy, x + dx) in positions for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))) >= 3
            for y, x in positions
        )
        large += len(component) >= 9
        elongated += max(h, w) >= 5
        template_by_anchor[anchor_color] = (component, (ay, ax), (y0, x0, h, w), carrier)
    unmatched = {color: points for color, points in markers.items() if color not in template_by_anchor}
    assert sum(len(v) for v in unmatched.values()) <= 4
    predicted = np.full(inp.shape, background, dtype=np.int8)
    boxes = []
    counts = {}
    for anchor_color, (component, anchor, bounds, carrier) in template_by_anchor.items():
        counts[anchor_color] = len(markers[anchor_color])
        assert 1 <= counts[anchor_color] <= 4
        for marker in markers[anchor_color]:
            top = marker[0] + bounds[0] - anchor[0]
            left = marker[1] + bounds[1] - anchor[1]
            box = (top, left, bounds[2], bounds[3])
            assert top >= 0 and left >= 0 and top + box[2] <= inp.shape[0] and left + box[3] <= inp.shape[1]
            assert all(not _overlap(box, old) for old in boxes)
            boxes.append(box)
            for y, x, color in component:
                yy, xx = marker[0] + y - anchor[0], marker[1] + x - anchor[1]
                assert predicted[yy, xx] == background
                predicted[yy, xx] = color
    assert np.array_equal(predicted, out)
    horizontal = vertical = False
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            horizontal |= (a[1] + a[3] == b[1] or b[1] + b[3] == a[1]) and max(a[0], b[0]) < min(a[0] + a[2], b[0] + b[2])
            vertical |= (a[0] + a[2] == b[0] or b[0] + b[2] == a[0]) and max(a[1], b[1]) < min(a[1] + a[3], b[1] + b[3])
    return {
        "background": background, "shape": inp.shape,
        "source_count": len(sources), "anchors": set(template_by_anchor),
        "carriers": {v[3] for v in template_by_anchor.values()},
        "counts": counts, "copy_count": len(boxes), "boxes": boxes,
        "unmatched": unmatched, "unmatched_repeated": any(len(v) >= 2 for v in unmatched.values()),
        "four_split": four_split, "diagonal_anchors": diagonal_anchors,
        "filled": filled, "filled_center": filled_center,
        "holes": holes, "interior_gaps": interior_gaps,
        "enclosed_holes": enclosed_holes, "branches": branches,
        "large": large, "elongated": elongated,
        "horizontal_touch": horizontal, "vertical_touch": vertical,
        "edge": any(y == 0 or x == 0 or y + h == inp.shape[0] or x + w == inp.shape[1] for y, x, h, w in boxes),
        "corner": any(y in (0, inp.shape[0] - h) and x in (0, inp.shape[1] - w) for y, x, h, w in boxes),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp); _check_grid(out)
            key_in = (inp.shape, inp.tobytes())
            key_pair = (key_in, out.shape, out.tobytes())
            assert key_in not in seen_inputs and key_pair not in seen_pairs
            seen_inputs.add(key_in); seen_pairs.add(key_pair)
            rows = tuple("".join(str(int(c)) for c in row) for row in inp)
            assert rows not in _ORIGINAL_INPUTS
            infos[split].append(_analyze_pair(inp, out))
    train_infos, test_infos = infos["train"], infos["test"]
    assert train_infos[0]["source_count"] == 2
    assert train_infos[0]["unmatched_repeated"]
    assert max(max(info["counts"].values()) for info in test_infos) <= max(max(info["counts"].values()) for info in train_infos)
    assert not any(len(info["carriers"]) > 1 for info in test_infos) or any(len(info["carriers"]) > 1 for info in train_infos)
    all_infos = train_infos + test_infos


def _same_puzzle(left, right):
    return all(
        np.array_equal(a[role], b[role])
        for split in ("train", "test")
        for a, b in zip(left[split], right[split])
        for role in ("input", "output")
    ) and all(len(left[s]) == len(right[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, backgrounds, palettes = set(), set(), set()
    signatures = set()
    metrics = {name: 0 for name in (
        "one_source", "four_source", "triple", "mixed", "diagonal", "diagonal_anchor",
        "filled", "filled_center", "holes", "interior_gaps", "enclosed_holes", "branches",
        "large", "elongated", "unmatched_zero", "unmatched_repeated",
        "horizontal_touch", "vertical_touch", "edge", "corner", "role_switch",
    )}
    within_sizes = within_backgrounds = within_carriers = 0
    within_source_counts = within_copy_counts = within_anchors = 0
    combinations = set()
    simple_tests = mixed_train = triple_train = 0
    sides = {side: 0 for side in ("top", "bottom", "left", "right")}
    no_contact = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"])); test_counts.add(len(puzzle["test"]))
        infos = [_analyze_pair(p["input"], p["output"]) for split in ("train", "test") for p in puzzle[split]]
        within_sizes += len({i["shape"] for i in infos}) > 1
        within_backgrounds += len({i["background"] for i in infos}) > 1
        within_carriers += len({tuple(sorted(i["carriers"])) for i in infos}) > 1
        within_source_counts += len({i["source_count"] for i in infos}) > 1
        within_copy_counts += len({i["copy_count"] for i in infos}) > 1
        within_anchors += len({tuple(sorted(i["anchors"])) for i in infos}) > 1
        mixed_train += any(len(i["carriers"]) > 1 for i in infos[:len(puzzle["train"])])
        triple_train += any(max(i["counts"].values()) >= 3 for i in infos[:len(puzzle["train"])])
        simple_tests += any(i["copy_count"] <= 3 for i in infos[len(puzzle["train"]):])
        anchor_union = set().union(*(i["anchors"] for i in infos))
        carrier_union = set().union(*(i["carriers"] for i in infos))
        metrics["role_switch"] += bool(anchor_union & carrier_union)
        for info in infos:
            sizes.add(info["shape"]); backgrounds.add(info["background"])
            palettes.add((info["background"], tuple(sorted(info["anchors"])), tuple(sorted(info["carriers"]))))
            metrics["one_source"] += info["source_count"] == 1
            metrics["four_source"] += info["source_count"] == 4
            metrics["triple"] += max(info["counts"].values()) >= 3
            metrics["mixed"] += len(info["carriers"]) >= 2
            metrics["diagonal"] += info["four_split"] > 0
            metrics["diagonal_anchor"] += info["diagonal_anchors"] > 0
            for name in ("filled", "filled_center", "holes", "interior_gaps", "enclosed_holes", "branches", "large", "elongated"):
                metrics[name] += info[name] > 0
            metrics["unmatched_zero"] += not info["unmatched"]
            metrics["unmatched_repeated"] += info["unmatched_repeated"]
            for name in ("horizontal_touch", "vertical_touch", "edge", "corner"):
                metrics[name] += info[name]
            height, width = info["shape"]
            sides["top"] += any(y == 0 for y, x, h, w in info["boxes"])
            sides["bottom"] += any(y + h == height for y, x, h, w in info["boxes"])
            sides["left"] += any(x == 0 for y, x, h, w in info["boxes"])
            sides["right"] += any(x + w == width for y, x, h, w in info["boxes"])
            no_contact += not (info["horizontal_touch"] or info["vertical_touch"])
            combinations.add((
                info["four_split"] > 0, len(info["carriers"]) >= 2,
                info["unmatched_repeated"],
                info["horizontal_touch"] or info["vertical_touch"], info["edge"],
            ))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes(), p["output"].tobytes()) for split in ("train", "test") for p in puzzle[split]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 30 and len(backgrounds) >= 8 and len(palettes) >= 100
    assert len(signatures) == 42
    assert within_sizes >= 32 and within_backgrounds >= 30 and within_carriers >= 30
    assert within_source_counts >= 30 and within_copy_counts >= 35
    assert within_anchors >= 35 and min(sides.values()) >= 20 and no_contact >= 40
    assert 4 <= 42 - within_backgrounds <= 15
    assert mixed_train >= 15 and triple_train >= 15 and simple_tests >= 8
    assert len(combinations) >= 24
    minimums = {
        "one_source": 8, "four_source": 8, "triple": 20, "mixed": 15,
        "diagonal": 20, "diagonal_anchor": 12, "filled": 20,
        "filled_center": 8, "holes": 20, "interior_gaps": 20,
        "enclosed_holes": 12,
        "branches": 20, "large": 20, "elongated": 15,
        "unmatched_zero": 10, "unmatched_repeated": 42,
        "horizontal_touch": 8, "vertical_touch": 8,
        "edge": 20, "corner": 5, "role_switch": 10,
    }
    assert all(metrics[name] >= bound for name, bound in minimums.items()), metrics
    for a, b in ((2, 1), (10, 1), (10, 4)):
        seed = 1200 + 10 * a + b
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
        assert _same_puzzle(puzzle, generate_puzzle(seed, a, b))
    for a, b in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2400 + (a or 0) * 10 + (b or 0), a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(9999, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
