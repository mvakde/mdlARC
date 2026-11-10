"""Noisy rectilinear partitions, recovered and encoded as sorted unary rows.

Color roles are local. Walls can be occluded by noise or another network, but
never silently erased to background. Equal counts have separated horizontal
extents, so left-edge, center, and right-edge order all agree.
"""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    "9e1f2810d5c9aef85bb746fabaf81001c28d20521e88dcbb48ccd44addfe6113",
    "919a7a088d27486bb4acd489814161473f4eaeea28886aa891bd3e16989855e5",
    "1cfd70c455602aeb8fe406cb67199b1b647467d74fa3e255dfab0d2c12ef60c0",
    "5af8bb746bfb5302d4a01373a204cd3e1f512cb84418b0d1c752d4eedd3c3a16",
}


def _as_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_pair_counts(rng, num_train, num_test):
    num_train = _as_count(num_train, "num_train")
    num_test = _as_count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _source_copy(pair):
    g, o = (np.asarray(pair[k], dtype=np.int8) for k in ("input", "output"))
    return hashlib.sha256(bytes(g.shape) + g.tobytes() + bytes(o.shape) + o.tobytes()).hexdigest() in _SOURCE_DIGESTS


def _rooms(mask):
    """Count bounded four-connected regions, including for edge-touching frames."""
    free = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros_like(free)
    rooms = []
    height, width = free.shape
    for r, c in np.argwhere(free):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells = []
        outside = False
        while stack:
            y, x = stack.pop()
            cells.append((y - 1, x - 1))
            outside |= y == 0 or x == 0 or y == height - 1 or x == width - 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and free[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if not outside:
            a = np.asarray(cells)
            lo, hi = a.min(axis=0), a.max(axis=0)
            # Recursive rectilinear partitions have rectangular elementary rooms.
            if len(cells) != int(np.prod(hi - lo + 1)):
                return None
            rooms.append(tuple(map(int, (*lo, *hi))))
    return rooms


def _walls(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top, left:right + 1] = True
        mask[bottom, left:right + 1] = True
        mask[top:bottom + 1, left] = True
        mask[top:bottom + 1, right] = True
    return mask


def _cycle_count(mask):
    """Planar Euler count, independent of the flood-fill counting algorithm.

    Adjacent wall pixels form a connected graph. Filled 2x2 blocks contribute
    unit faces with no room interior, and must be removed from its cycle rank.
    """
    edges = int(np.count_nonzero(mask[:, :-1] & mask[:, 1:]))
    edges += int(np.count_nonzero(mask[:-1, :] & mask[1:, :]))
    unit_faces = int(np.count_nonzero(mask[:-1, :-1] & mask[1:, :-1]
                                     & mask[:-1, 1:] & mask[1:, 1:]))
    return edges - int(mask.sum()) + 1 - unit_faces


def _recover_network(grid, color, background):
    """Recover actual segments, with no continuation across background cells.

    Adjacent support identifies line coordinates. Elementary intervals between
    those coordinates can be repaired only when they contain structural evidence
    and every missing pixel is visibly occupied by another color. A deliberately
    absent wall therefore has a different observation from an occluded wall.
    """
    visible = grid == color
    positions = np.argwhere(visible)
    if len(positions) < 8:
        return None
    top, left = map(int, positions.min(axis=0))
    bottom, right = map(int, positions.max(axis=0))
    hs = np.flatnonzero(np.any(visible[:, :-1] & visible[:, 1:], axis=1)).tolist()
    vs = np.flatnonzero(np.any(visible[:-1, :] & visible[1:, :], axis=0)).tolist()
    if top not in hs or bottom not in hs or left not in vs or right not in vs:
        return None
    mask = visible.copy()
    for y in hs:
        for x1, x2 in zip(vs, vs[1:]):
            line = grid[y, x1:x2 + 1]
            if np.all(line != background) and np.any(line == color):
                mask[y, x1:x2 + 1] = True
    for x in vs:
        for y1, y2 in zip(hs, hs[1:]):
            line = grid[y1:y2 + 1, x]
            if np.all(line != background) and np.any(line == color):
                mask[y1:y2 + 1, x] = True
    if not (mask[top, left:right + 1].all() and mask[bottom, left:right + 1].all()
            and mask[top:bottom + 1, left].all() and mask[top:bottom + 1, right].all()):
        return None
    # Crop the flood fill for speed; retain coordinates in the full input.
    local = _rooms(mask[top:bottom + 1, left:right + 1])
    if not local:
        return None
    rooms = [(a + top, b + left, c + top, d + left) for a, b, c, d in local]
    boundaries = [(a - 1, b - 1, c + 1, d + 1) for a, b, c, d in rooms]
    if not np.array_equal(mask, _walls(grid.shape, boundaries)):
        return None
    # Retain visible background witnesses for intentionally absent continuations.
    # Entirely hidden extra dividers between existing line coordinates are banned.
    for y in hs:
        for a, b in zip(vs, vs[1:]):
            if b - a >= 2 and mask[y, a] and mask[y, b] and not mask[y, a:b + 1].all():
                if not np.any(grid[y, a + 1:b] == background):
                    return None
    for x in vs:
        for a, b in zip(hs, hs[1:]):
            if b - a >= 2 and mask[a, x] and mask[b, x] and not mask[a:b + 1, x].all():
                if not np.any(grid[a + 1:b, x] == background):
                    return None
    return {"color": int(color), "count": len(rooms), "mask": mask,
            "rooms": rooms, "box": (top, left, bottom, right)}


def _decode_input(grid):
    colors, frequencies = np.unique(grid, return_counts=True)
    assert len(colors) >= 3
    ranking = np.argsort(frequencies)
    assert frequencies[ranking[-1]] > frequencies[ranking[-2]], "ambiguous background"
    background = int(colors[ranking[-1]])
    networks, noise = [], []
    for color in colors:
        if color == background:
            continue
        network = _recover_network(grid, int(color), background)
        if network is None:
            noise.append(int(color))
        else:
            networks.append(network)
    assert len(noise) == 1 and networks, "roles must be uniquely visible"
    networks.sort(key=lambda n: (n["count"], n["box"][1]))
    return background, noise[0], networks


def _output(noise, networks):
    ordered = sorted(networks, key=lambda n: (n["count"], n["box"][1]))
    result = np.full((len(ordered), max(n["count"] for n in ordered)), noise, dtype=np.int8)
    for row, network in enumerate(ordered):
        result[row, :network["count"]] = network["color"]
    return result


def _capacity(box):
    t, l, b, r = box
    return ((b - t) // 2) * ((r - l) // 2)


def _offsets(parts, span, rng):
    gaps = np.full(parts, 2, dtype=int)
    for _ in range(span - 2 * parts):
        gaps[int(rng.integers(parts))] += 1
    return [0] + list(map(int, np.cumsum(gaps)))


def _partition(rng, height, width, count, style):
    """Sample segment endpoints by successively splitting individual rooms."""
    outer = (0, 0, height - 1, width - 1)
    if _capacity(outer) < count:
        return None
    if style in ("strip", "grid"):
        factors = [(r, count // r) for r in range(1, count + 1)
                   if count % r == 0 and 2 * r < height and 2 * (count // r) < width
                   and (style == "grid" or r == 1 or r == count)]
        if not factors:
            # A long strip may not fit a crowded canvas even though a mixed
            # partition with the same count does. Geometry is a pair parameter.
            return _partition(rng, height, width, count, "partial")
        nr, nc = factors[int(rng.integers(len(factors)))]
        ys, xs = _offsets(nr, height - 1, rng), _offsets(nc, width - 1, rng)
        boxes = [(ys[i], xs[j], ys[i + 1], xs[j + 1]) for i in range(nr) for j in range(nc)]
    else:
        boxes = [outer]
        first_axis = None
        for step in range(count - 1):
            choices = []
            for i, (t, l, b, r) in enumerate(boxes):
                for axis, start, end in ((0, t, b), (1, l, r)):
                    # A perpendicular second cut creates a real T-junction.
                    if style == "partial" and count >= 3 and step == 1 and axis == first_axis:
                        continue
                    for cut in range(start + 2, end - 1):
                        halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                        total_capacity = sum(_capacity(x) for j, x in enumerate(boxes) if j != i)
                        total_capacity += sum(_capacity(x) for x in halves)
                        if total_capacity >= count:
                            choices.append((i, axis, halves))
            if not choices:
                return None
            i, axis, halves = choices[int(rng.integers(len(choices)))]
            if step == 0:
                first_axis = axis
            boxes[i:i + 1] = halves
    mask = _walls((height, width), boxes)
    if style == "partial" and count >= 3:
        ys = {y for t, _, b, _ in boxes for y in (t, b)}
        xs = {x for _, l, _, r in boxes for x in (l, r)}
        # Later cuts can accidentally complete a lattice. A partial-divider
        # demonstration must still have an absent continuation in its final mask.
        if all(mask[y, :].all() for y in ys) and all(mask[:, x].all() for x in xs):
            return None
    return mask


def _tie_clear(networks):
    for i, a in enumerate(networks):
        for b in networks[i + 1:]:
            if a["count"] == b["count"]:
                if not (a["box"][3] < b["box"][1] or b["box"][3] < a["box"][1]):
                    return False
    return True


def _overlap(networks):
    return bool(np.any(sum(n["mask"].astype(int) for n in networks) > 1))


def _dimensions(rng, number, size_mode, fallback):
    if fallback:
        return 30, 30
    low = 7 if number == 1 else 13 if number <= 3 else 21
    if size_mode == "large":
        low = max(low, 26)
    high = 25 if size_mode == "source_small" else 23 if size_mode == "small" and number <= 3 else 31
    return int(rng.integers(low, high)), int(rng.integers(low, high))


def _layout(rng, shape, counts, colors, styles, disjoint, order_witness, fallback):
    height, width = shape
    number = len(counts)
    # Disjoint placement uses shuffled cells; ordinary overlap placement uses
    # independent boxes and layers. Tied networks always occupy distinct columns.
    columns = min(number, 3)
    rows = (number + columns - 1) // columns
    slots = [(i // columns, i % columns) for i in range(number)]
    for _ in range(40):
        rng.shuffle(slots)
        if all(counts[i] != counts[j] or slots[i][1] != slots[j][1]
               for i in range(number) for j in range(i)):
            break
    else:
        return None
    networks = []
    for index, (count, color, style) in enumerate(zip(counts, colors, styles)):
        accepted = None
        for _ in range(90):
            if disjoint:
                sr, sc = slots[index]
                y0, y1 = sr * height // rows, (sr + 1) * height // rows - 1
                x0, x1 = sc * width // columns, (sc + 1) * width // columns - 1
            else:
                y0, y1, x0, x1 = 0, height - 1, 0, width - 1
            maxh, maxw = y1 - y0 + 1, x1 - x0 + 1
            if min(maxh, maxw) < 3:
                continue
            if fallback:
                bh, bw = maxh, maxw
                # Guaranteed spacious overlapping two-network demonstration.
                if order_witness:
                    bh, bw = 20, 18
            else:
                caph = maxh if number <= 3 or disjoint else min(maxh, 16)
                capw = maxw if number <= 3 or disjoint else min(maxw, 16)
                bh, bw = int(rng.integers(3, caph + 1)), int(rng.integers(3, capw + 1))
            if bh > maxh or bw > maxw:
                continue
            local = _partition(rng, bh, bw, count, style)
            if local is None:
                continue
            top, left = int(rng.integers(y0, y1 - bh + 2)), int(rng.integers(x0, x1 - bw + 2))
            if fallback and order_witness:
                top, left = (0, 0) if index == 0 else (7, 10)
            bottom, right = top + bh - 1, left + bw - 1
            mask = np.zeros(shape, dtype=bool)
            mask[top:bottom + 1, left:right + 1] = local
            candidate = {"color": int(color), "count": int(count), "box": (top, left, bottom, right), "mask": mask}
            if not _tie_clear(networks + [candidate]):
                continue
            if order_witness and index == 1:
                if networks[0]["box"][1] >= left or not np.any(networks[0]["mask"] & mask):
                    continue
            if networks:
                union = np.logical_or.reduce([n["mask"] for n in networks])
                if np.count_nonzero(mask & union) > 0.20 * mask.sum():
                    continue
            accepted = candidate
            break
        if accepted is None:
            return None
        networks.append(accepted)
    return networks


def _same_recovery(grid, background, noise, networks):
    try:
        bg, ns, recovered = _decode_input(grid)
    except AssertionError:
        return False
    if bg != background or ns != noise or len(recovered) != len(networks):
        return False
    intended = {n["color"]: n for n in networks}
    return all(n["color"] in intended and np.array_equal(n["mask"], intended[n["color"]]["mask"])
               and n["count"] == intended[n["color"]]["count"] for n in recovered)


def _compose(rng, shape, background, noise, networks, require_damage, fallback):
    clean = np.full(shape, background, dtype=np.int8)
    for i in rng.permutation(len(networks)):
        n = networks[int(i)]
        clean[n["mask"]] = n["color"]
    union = np.logical_or.reduce([n["mask"] for n in networks])
    # The background remains locally recognizable; no color is reserved at edges.
    if np.count_nonzero(clean == background) < 0.36 * clean.size:
        return None
    for attempt in range(16):
        grid = clean.copy()
        for n in networks:
            if rng.random() < (0.85 if require_damage else 0.60):
                candidates = np.argwhere(grid == n["color"])
                rate = float(rng.uniform(0.015, 0.16)) / (1 + attempt // 6)
                amount = max(1, int(len(candidates) * rate))
                indices = rng.choice(len(candidates), size=amount, replace=False)
                for y, x in candidates[indices]:
                    grid[y, x] = noise
        density = float(rng.uniform(0.003, 0.145))
        if fallback:
            density = 0.012
        eligible = grid == background
        # Some scenes happen to have clean borders; other scenes scatter anywhere.
        if rng.random() < 0.35:
            eligible[[0, -1], :] = False
            eligible[:, [0, -1]] = False
        sites = np.argwhere(eligible)
        if not len(sites):
            continue
        amount = min(len(sites), max(1, int(density * grid.size)))
        weights = np.ones(len(sites))
        if rng.random() < 0.5:
            center = sites[int(rng.integers(len(sites)))]
            weights += 14 * np.exp(-np.sum((sites - center) ** 2, axis=1) / float(rng.uniform(3, 30)))
        for y, x in sites[rng.choice(len(sites), size=amount, replace=False, p=weights / weights.sum())]:
            grid[y, x] = noise
        if require_damage and not np.any(union & (grid == noise)):
            continue
        if _same_recovery(grid, background, noise, networks):
            return grid
    return None


def _counts(rng, maximum, ties, witness=None, extrapolate=False):
    if witness == "order":
        return [maximum, int(rng.integers(1, maximum))]
    if witness == "tie":
        value = int(rng.integers(1, maximum + 1))
        group = int(rng.choice([2, 2, 3]))
        result = [value] * group
        if rng.random() < 0.5:
            options = [x for x in range(1, maximum + 1) if x != value]
            result.append(int(rng.choice(options)))
        return result
    number = int(rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 6]))
    if not ties:
        number = min(number, maximum)
        result = list(map(int, rng.choice(np.arange(1, maximum + 1), size=number, replace=False)))
    else:
        result = []
        for _ in range(number):
            options = [x for x in range(1, maximum + 1) if result.count(x) < 3]
            result.append(int(rng.choice(options)))
    if extrapolate and maximum not in result:
        result[0] = maximum
    return result


def _make_pair(rng, parameters, regime, witness=None):
    assert regime in ("source_train", "target_test")
    maximum = parameters["maximum"]
    extrapolate = regime == "target_test" and parameters["count_mode"] == "extrapolate"
    if extrapolate:
        maximum += int(rng.integers(1, 4))
    counts = _counts(rng, maximum, parameters["ties"], witness, extrapolate)
    role_mode = parameters["role_mode"]
    background, noise = parameters["roles"]
    if role_mode == "noise":
        noise = int(rng.choice([x for x in range(10) if x != background]))
    elif role_mode == "local":
        background, noise = map(int, rng.choice(10, size=2, replace=False))
    palette = [c for c in range(10) if c not in (background, noise)]
    colors = list(map(int, rng.choice(palette, size=len(counts), replace=False)))
    styles = [str(rng.choice(["strip", "grid", "partial", "partial", "partial"])) for _ in counts]
    if witness == "order":
        styles[0] = "partial"
    if regime == "target_test" and parameters["count_mode"] == "recombine":
        styles = ["partial"] * len(counts)
    size_mode = parameters["size_mode"] if regime == "target_test" else "source_small" if parameters["size_mode"] == "large" else "shared"
    # Counts and roles survive retries. Geometry can adapt to available space;
    # fallback never changes the requested rule or compartment count regime.
    for attempt in range(140):
        fallback = attempt >= 100
        shape = _dimensions(rng, len(counts), size_mode, fallback)
        if not fallback and size_mode == "matched":
            shapes = parameters["train_shapes"]
            shape = shapes[int(rng.integers(len(shapes)))]
        disjoint = witness == "tie" or (witness != "order" and (fallback or rng.random() < 0.40))
        networks = _layout(rng, shape, counts, colors, styles, disjoint, witness == "order", fallback)
        if networks is None:
            continue
        grid = _compose(rng, shape, background, noise, networks, witness == "order", fallback)
        if grid is None:
            continue
        # Ground truth comes from the clean segment union, via actual bounded
        # regions, and is checked against reconstruction from the returned grid.
        for n in networks:
            t, l, b, r = n["box"]
            rooms = _rooms(n["mask"][t:b + 1, l:r + 1])
            assert rooms is not None and len(rooms) == n["count"]
        pair = {"input": grid, "output": _output(noise, networks)}
        if not _source_copy(pair):
            return pair
    return _fallback_pair(counts, colors, background, noise, witness)


def _fallback_pair(counts, colors, background, noise, witness):
    """Finite, non-random safety construction for tight packing requests."""
    shape = (30, 30)
    networks = []
    if witness == "order":
        # A partial left network and a later, lower-count frame cross at two
        # isolated pixels. Neither an entire segment nor a corner is obscured.
        boxes = [(2, 1, 20, 17), (10, 12, 26, 27)]
        for i, (box, count, color) in enumerate(zip(boxes, counts, colors)):
            t, l, b, r = box
            if i == 0:
                cut = l + 6
                ys = [t + (b - t) * k // (count - 1) for k in range(count)]
                rooms = [(ys[k], l, ys[k + 1], cut) for k in range(count - 1)] + [(t, cut, b, r)]
            else:
                ys = [t + (b - t) * k // count for k in range(count + 1)]
                rooms = [(ys[k], l, ys[k + 1], r) for k in range(count)]
            networks.append(dict(box=box, count=count, color=color, mask=_walls(shape, rooms)))
    else:
        # Grouping equal counts before cyclic column assignment separates every
        # tie, even two different groups of three in a six-network scene.
        order = sorted(range(len(counts)), key=lambda i: counts[i])
        for slot, i in enumerate(order):
            t, l = (slot // 3) * 15 + 1, (slot % 3) * 10 + 1
            b, r = t + 12, l + 7
            count = counts[i]
            if count == 1:
                rooms = [(t, l, b, r)]
            else:
                left_count = min(6, count - 1)
                right_count = count - left_count
                rooms = []
                for x1, x2, amount in ((l, l + 3, left_count), (l + 3, r, right_count)):
                    ys = [t + 12 * k // amount for k in range(amount + 1)]
                    rooms.extend((ys[k], x1, ys[k + 1], x2) for k in range(amount))
            networks.append(dict(box=(t, l, b, r), count=count, color=colors[i], mask=_walls(shape, rooms)))
    grid = np.full(shape, background, dtype=np.int8)
    for n in networks:
        grid[n["mask"]] = n["color"]
    grid[28, 15] = noise
    if witness == "order":
        grid[2, 3] = noise
    assert _same_recovery(grid, background, noise, networks)
    assert _tie_clear(networks)
    return {"input": grid, "output": _output(noise, networks)}


def _analysis(pair):
    background, noise, networks = _decode_input(pair["input"])
    assert np.array_equal(pair["output"], _output(noise, networks)), "incorrect room encoding"
    assert _tie_clear(networks), "tied networks need clear horizontal order"
    assert all(_cycle_count(n["mask"]) == n["count"] for n in networks)
    return background, noise, networks


def _training_evidence(train):
    analyses = [_analysis(p) for p in train]
    order, padding, tie, partial, damage, overlap = False, False, False, False, False, False
    for pair, (_, noise, networks) in zip(train, analyses):
        counts = [n["count"] for n in networks]
        padding |= min(counts) < max(counts)
        spatial = sorted(networks, key=lambda n: n["box"][1])
        order |= [n["count"] for n in spatial] != sorted(counts)
        tie |= len(set(counts)) < len(counts)
        overlap |= _overlap(networks)
        for n in networks:
            damage |= bool(np.any(n["mask"] & (pair["input"] == noise)))
            partial |= _topology(n)["t_junctions"] > 0
    return dict(order=order, padding=padding, tie=tie, partial=partial, damage=damage, overlap=overlap)


def solve_from_training(train_pairs, test_input):
    """Prediction uses only demonstration pairs and the new input, never a label."""
    evidence = _training_evidence(train_pairs)
    assert evidence["order"] and evidence["padding"]
    _, noise, networks = _decode_input(test_input)
    counts = [n["count"] for n in networks]
    if len(set(counts)) < len(counts):
        assert evidence["tie"]
    if any(_topology(n)["t_junctions"] for n in networks):
        assert evidence["partial"]
    if _overlap(networks):
        assert evidence["overlap"]
    if any(np.any(n["mask"] & (test_input == noise)) for n in networks):
        assert evidence["damage"]
    assert _tie_clear(networks)
    return _output(noise, networks)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert not _source_copy(pair)
            assert not np.array_equal(pair["input"], pair["output"])
            _analysis(pair)
    evidence = _training_evidence(puzzle["train"])
    assert all(evidence[k] for k in ("order", "padding", "partial", "damage", "overlap"))
    for pair in puzzle["test"]:
        prediction = solve_from_training(puzzle["train"], pair["input"].copy())
        assert np.array_equal(prediction, pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ne = _resolve_pair_counts(rng, num_train, num_test)
    parameters = {"roles": tuple(map(int, rng.choice(10, size=2, replace=False))),
                  "role_mode": str(rng.choice(["fixed", "noise", "local", "local"])),
                  "maximum": int(rng.integers(3, 7)), "ties": bool(rng.random() < 0.75),
                  "count_mode": str(rng.choice(["shared", "shared", "extrapolate", "recombine"])),
                  "size_mode": str(rng.choice(["shared", "matched", "large", "small"])),
                  "train_shapes": []}
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, number, regime in (("train", nt, "source_train"), ("test", ne, "target_test")):
        for i in range(number):
            witness = "order" if split == "train" and i == 0 else "tie" if split == "train" and i == 1 and parameters["ties"] else None
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(8):
                pair = _make_pair(pair_rng, parameters, regime, witness)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        parameters["train_shapes"].append(pair["input"].shape)
                    break
            else:
                raise AssertionError("duplicate input after bounded retries")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _segment_edges(network):
    """Unit edges of recovered room boundaries, excluding incidental adjacency."""
    mask = network["mask"]
    rooms = network.get("rooms")
    if rooms is None:
        rooms = _rooms(mask)
    horizontal = np.zeros((mask.shape[0], mask.shape[1] - 1), dtype=bool)
    vertical = np.zeros((mask.shape[0] - 1, mask.shape[1]), dtype=bool)
    for a, b, c, d in rooms:
        horizontal[a - 1, b - 1:d + 1] = True
        horizontal[c + 1, b - 1:d + 1] = True
        vertical[a - 1:c + 1, b - 1] = True
        vertical[a - 1:c + 1, d + 1] = True
    return horizontal, vertical, rooms


def _topology(network):
    mask = network["mask"]
    horizontal, vertical, rooms = _segment_edges(network)
    degree = np.zeros(mask.shape, dtype=int)
    degree[:, :-1] += horizontal
    degree[:, 1:] += horizontal
    degree[:-1, :] += vertical
    degree[1:, :] += vertical
    sizes = [(c - a + 1, d - b + 1) for a, b, c, d in rooms]
    top, left, bottom, right = network["box"]
    ends = {"left": [], "right": [], "up": [], "down": []}
    for y, x in np.argwhere(degree == 3):
        if not (top < y < bottom and left < x < right):
            continue
        if vertical[y - 1, x] and vertical[y, x]:
            ends["left" if horizontal[y, x - 1] else "right"].append((int(y), int(x)))
        if horizontal[y, x - 1] and horizontal[y, x]:
            ends["up" if vertical[y - 1, x] else "down"].append((int(y), int(x)))
    staggered = (any(x1 == x2 and y1 != y2 for y1, x1 in ends["left"] for y2, x2 in ends["right"])
                 or any(y1 == y2 and x1 != x2 for y1, x1 in ends["up"] for y2, x2 in ends["down"]))
    return {"t_junctions": int(np.count_nonzero(degree[top + 1:bottom, left + 1:right] == 3)),
            "crossings": int(np.count_nonzero(mask & (degree == 4))),
            "staggered": staggered,
            "room_sizes": sizes, "horizontal": bool(horizontal[top + 1:bottom].any()),
            "vertical": bool(vertical[:, left + 1:right].any())}


def measure_puzzle(puzzle):
    """Review metrics derived from returned pixels, not sampling parameters."""
    result = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            bg, noise, networks = _analysis(pair)
            topology = [_topology(n) for n in networks]
            union = np.logical_or.reduce([n["mask"] for n in networks])
            damaged = [bool(np.any(n["mask"] & (pair["input"] == noise))) for n in networks]
            invisible = [bool(np.any(n["mask"] & (pair["input"] != n["color"]))) for n in networks]
            border = np.zeros_like(union)
            border[[0, -1], :] = True
            border[:, [0, -1]] = True
            junction_damage = corner_damage = max_gap = 0
            for n in networks:
                he, ve, _ = _segment_edges(n)
                deg = np.zeros(n["mask"].shape, dtype=int)
                deg[:, :-1] += he
                deg[:, 1:] += he
                deg[:-1, :] += ve
                deg[1:, :] += ve
                junction_damage += int(np.count_nonzero(n["mask"] & (deg >= 3) & (pair["input"] == noise)))
                h_arms = np.pad(he, ((0, 0), (0, 1))) | np.pad(he, ((0, 0), (1, 0)))
                v_arms = np.pad(ve, ((0, 1), (0, 0))) | np.pad(ve, ((1, 0), (0, 0)))
                corner_damage += int(np.count_nonzero((deg == 2) & h_arms & v_arms & (pair["input"] == noise)))
                missing = n["mask"] & (pair["input"] != n["color"])
                for field in (missing, missing.T):
                    for line in field:
                        runs = np.diff(np.flatnonzero(np.r_[True, ~line, True])) - 1
                        max_gap = max(max_gap, int(runs.max()))
            counts = [n["count"] for n in networks]
            dimensions = [size for t in topology for size in t["room_sizes"]]
            result.append({"split": split, "shape": list(pair["input"].shape), "background": bg, "noise": noise,
                           "networks": len(networks), "counts": counts, "maximum": max(counts),
                           "ties": len(counts) - len(set(counts)), "largest_tie": max(counts.count(x) for x in counts),
                           "padding": min(counts) < max(counts), "partial_networks": sum(t["t_junctions"] > 0 for t in topology),
                           "t_junctions": sum(t["t_junctions"] for t in topology),
                           "grid_networks": sum(t["horizontal"] and t["vertical"] and not t["t_junctions"] for t in topology),
                           "strip_networks": sum(t["horizontal"] != t["vertical"] for t in topology),
                           "staggered_networks": sum(t["staggered"] for t in topology),
                           "crossings": sum(t["crossings"] for t in topology), "room_sizes": dimensions,
                           "horizontal": any(t["horizontal"] for t in topology), "vertical": any(t["vertical"] for t in topology),
                           "unequal_rooms": any(len(set(t["room_sizes"])) > 1 for t in topology),
                           "overlap": _overlap(networks), "damaged_networks": sum(damaged),
                           "intact_networks": len(networks) - sum(invisible), "junction_damage": junction_damage,
                           "corner_damage": corner_damage, "max_occlusion_run": max_gap,
                           "noise_density": float(np.mean(pair["input"] == noise)),
                           "edge_frame": bool(np.any(union & border)), "border_noise": int(np.count_nonzero((pair["input"] == noise) & border)),
                           "background_noise_cluster": bool(np.any((pair["input"][:, :-1] == noise) & (pair["input"][:, 1:] == noise) & ~union[:, :-1] & ~union[:, 1:])),
                           "equal_left_unequal_count": any(a["box"][1] == b["box"][1] and a["count"] != b["count"] for a in networks for b in networks),
                           "large_single_beside_compact_many": any(a["count"] == 1 and b["count"] >= 3 and
                               (a["box"][2] - a["box"][0] - 1) * (a["box"][3] - a["box"][1] - 1) >= 100 and
                               (a["box"][2] - a["box"][0]) * (a["box"][3] - a["box"][1]) >
                               2 * (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])
                               for a in networks for b in networks),
                           "area_count_inversion": any(a["count"] < b["count"] and
                               (a["box"][2] - a["box"][0]) * (a["box"][3] - a["box"][1]) >
                               (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])
                               for a in networks for b in networks)})
    return result


def self_test() -> None:
    records, puzzles, train_counts, test_counts = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        records.extend(measure_puzzle(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    for seed in (0, 8):
        again = generate_puzzle(seed)
        assert all(_grid_key(a[k]) == _grid_key(b[k]) for s in again for a, b in zip(again[s], puzzles[seed][s]) for k in a)
    assert _grid_key(puzzles[0]["train"][0]["input"]) != _grid_key(puzzles[1]["train"][0]["input"])
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(100 + i, nt, ne)
        assert nt is None or len(p["train"]) == nt
        assert ne is None or len(p["test"]) == ne
    # Regression: successive cuts once completed every partial demonstration.
    generate_puzzle(815066666424)
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Exercise the finite fallback independently of how often sampling needs it.
    for high in range(3, 7):
        for low in range(1, high):
            p = _fallback_pair([high, low], [1, 2], 0, 5, "order")
            _analysis(p)
    for count in range(1, 10):
        _analysis(_fallback_pair([count] * 3, [1, 2, 3], 0, 5, None))
    # Rotation/reflection keep unary counts and local color meanings. Use
    # tie-free cases, since reflecting a tie intentionally reverses its order.
    for puzzle in puzzles[:4]:
        for pair in puzzle["train"] + puzzle["test"]:
            _, noise, ns = _decode_input(pair["input"])
            if len({n["count"] for n in ns}) == len(ns):
                for transformed in (pair["input"].T, np.rot90(pair["input"]), np.fliplr(pair["input"])):
                    _, local_noise, recovered = _decode_input(transformed)
                    assert local_noise == noise
                    assert np.array_equal(_output(local_noise, recovered), pair["output"])
                colors = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
                _, local_noise, recovered = _decode_input(colors[pair["input"]])
                assert np.array_equal(_output(local_noise, recovered), colors[pair["output"]])
    assert len({tuple(r["shape"]) for r in records}) >= 70
    assert len({tuple(r["counts"]) for r in records}) >= 50
    assert set(range(1, 7)).issubset({r["networks"] for r in records})
    assert {22, 29, 30}.issubset({r["shape"][0] for r in records})
    assert min(min(r["shape"]) for r in records) <= 10
    assert any(r["shape"][0] >= 2 * r["shape"][1] for r in records)
    assert any(r["shape"][1] >= 2 * r["shape"][0] for r in records)
    assert any(r["largest_tie"] >= 3 for r in records)
    assert any(not r["padding"] and r["networks"] > 1 for r in records)
    assert any(1 not in r["counts"] for r in records)
    for key in ("overlap", "edge_frame", "unequal_rooms", "area_count_inversion", "background_noise_cluster"):
        assert sum(r[key] for r in records) >= 10, key
    assert any(r["border_noise"] == 0 for r in records)
    assert any(r["noise_density"] > 0.086 for r in records)
    assert any(r["junction_damage"] for r in records)
    assert any(r["corner_damage"] for r in records)
    assert any(r["max_occlusion_run"] >= 3 for r in records)
    assert any(r["equal_left_unequal_count"] for r in records)
    assert any(r["large_single_beside_compact_many"] for r in records)
    assert any(r["damaged_networks"] and r["intact_networks"] for r in records)
    assert any(not r["overlap"] and not r["damaged_networks"] for r in records)
    assert any(r["partial_networks"] and r["overlap"] and r["unequal_rooms"] and r["damaged_networks"] for r in records)
    sizes = [size for r in records for size in r["room_sizes"]]
    assert min(min(s) for s in sizes) == 1 and max(max(s) for s in sizes) >= 20
    assert sum(r["partial_networks"] > 0 for r in records) >= 80
    assert any(r["crossings"] > 0 for r in records)
    assert sum(r["grid_networks"] for r in records) >= 10
    assert sum(r["strip_networks"] for r in records) >= 30
    assert sum(r["staggered_networks"] for r in records) >= 10
    assert any(r["horizontal"] and r["vertical"] for r in records)
    role_changes = count_extrapolation = shared_counts = smaller_tests = within_partial = 0
    for p in puzzles:
        rs = measure_puzzle(p)
        tr, te = [r for r in rs if r["split"] == "train"], [r for r in rs if r["split"] == "test"]
        role_changes += len({(r["background"], r["noise"]) for r in rs}) > 1
        count_extrapolation += max(r["maximum"] for r in te) > max(r["maximum"] for r in tr)
        shared_counts += max(r["maximum"] for r in te) <= max(r["maximum"] for r in tr)
        smaller_tests += min(np.prod(r["shape"]) for r in te) < max(np.prod(r["shape"]) for r in tr)
        within_partial += len({bool(r["partial_networks"]) for r in rs}) > 1
    assert role_changes >= 15 and count_extrapolation >= 5 and shared_counts >= 5
    assert smaller_tests >= 10 and within_partial >= 15
