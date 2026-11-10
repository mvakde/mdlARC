"""Compositional generators for seeded, one-cell-inset ARC networks."""

import hashlib
import numpy as np


_SOURCE_PAIR_HASHES = frozenset({
    "e3add08a22b3ca5900d6d6b4506e76f3cda59bbbaefbb92f02acc92db43a8879",
    "68b623662853f6d3fb608d663444830b21a588f0b0462c6a2c9f4f76f6a36f54",
    "8b8520dac93b48787cfa17dee89a896214f2213d84feeb262aa396664d03fe66",
    "f5405b1575c839a5024e8c72f1e99438e4d4005ca6710465016334337ac37d1b",
})


def _count(name, value, lo, hi):
    if value is not None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")


def _resolve_counts(rng, seed, num_train, num_test):
    _count("num_train", num_train, 2, 10)
    _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        nt = 1 + seed % 4
        nr = nt + 1 + (seed // 4) % (10 - nt)
    elif num_train is None:
        nt = int(num_test)
        nr = int(rng.integers(max(2, nt + 1), 11))
    elif num_test is None:
        nr = int(num_train)
        nt = int(rng.integers(1, min(4, nr - 1) + 1))
    else:
        nr, nt = int(num_train), int(num_test)
    if nr <= nt:
        raise ValueError("num_train must exceed num_test")
    return nr, nt


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        todo = [(y, x)]
        cells = []
        while todo:
            a, b = todo.pop()
            cells.append((a, b))
            for c, d in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if 0 <= c < h and 0 <= d < w and mask[c, d] and not seen[c, d]:
                    seen[c, d] = True
                    todo.append((c, d))
        found.append(cells)
    return found


def _interior(mask):
    result = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) < 3:
        return result
    core = result[1:-1, 1:-1]
    core[:] = True
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            core &= mask[1 + dy:mask.shape[0] - 1 + dy,
                         1 + dx:mask.shape[1] - 1 + dx]
    return result


def _roles_and_components(inp):
    """Parse roles from the input alone, including edge-touching pads."""
    colors = [int(v) for v in np.unique(inp)]
    matches = []
    for bg in colors:
        mask = inp != bg
        if int(mask.sum()) < 18:
            continue
        regions = _components(mask)
        if not regions or len(regions) > 5:
            continue
        for structure in colors:
            if structure == bg:
                continue
            markers = []
            interiors = []
            valid = True
            for cells in regions:
                cmask = np.zeros_like(mask)
                yy, xx = zip(*cells)
                cmask[yy, xx] = True
                exceptional = [(y, x) for y, x in cells if inp[y, x] != structure]
                if len(exceptional) != 1:
                    valid = False
                    break
                y, x = exceptional[0]
                inner = _interior(cmask)
                if not inner[y, x] or int(inner.sum()) < 2:
                    valid = False
                    break
                markers.append(int(inp[y, x]))
                interiors.append(inner)
            if valid:
                matches.append((bg, structure, regions, markers, interiors))
    if len(matches) != 1:
        raise AssertionError(f"input roles are ambiguous: {len(matches)} parses")
    return matches[0]


def _predict(inp):
    bg, structure, regions, markers, interiors = _roles_and_components(inp)
    out = inp.copy()
    for color, inner in zip(markers, interiors):
        out[inner] = color
    return out


def _axis(rng, length, slots, large=False, long_gap=False):
    sizes = [3] * slots
    gaps = [1] * (slots - 1)
    special = int(rng.integers(slots)) if large else None
    if large:
        sizes[special] += large
    gap_special = int(rng.integers(slots - 1)) if long_gap and slots > 1 else None
    if gap_special is not None:
        gaps[gap_special] += min(3, length - sum(sizes) - sum(gaps))
    budget = length - sum(sizes) - sum(gaps)
    if budget < 0:
        return None
    spending = int(rng.integers(max(0, budget // 3), budget + 1))
    for _ in range(spending):
        options = [(0, i) for i, v in enumerate(sizes) if v < (12 if i == special else 6)]
        options += [(1, i) for i, v in enumerate(gaps) if v < 6]
        if not options:
            break
        kind, index = options[int(rng.integers(len(options)))]
        (sizes if kind == 0 else gaps)[index] += 1
    span = sum(sizes) + sum(gaps)
    slack = length - span
    start = int(rng.choice((0, slack, int(rng.integers(slack + 1)))))
    starts = []
    at = start
    for i, size in enumerate(sizes):
        starts.append(at)
        at += size + (gaps[i] if i < len(gaps) else 0)
    lanes = [starts[i] + int(rng.integers(1, sizes[i] - 1)) for i in range(slots)]
    return sizes, starts, lanes, special, gaps


def _slot_labels(rng, rows, cols, components, total, required=None):
    """Grow disjoint connected sets of occupied slots on a common grid."""
    all_slots = [(r, c) for r in range(rows) for c in range(cols)]
    targets = [2] * components
    for _ in range(total - 2 * components):
        choices = [k for k in range(components) if targets[k] < 10]
        if not choices:
            break
        k = int(rng.choice(choices))
        targets[k] += 1
    for _ in range(50):
        labels = np.full((rows, cols), -1, dtype=np.int8)
        available = all_slots.copy()
        if required is not None:
            seeds = [required]
            available.remove(required)
        else:
            seeds = []
        while len(seeds) < components:
            if not available:
                break
            choice = available[int(rng.integers(len(available)))]
            seeds.append(choice)
            available.remove(choice)
        for k, (r, c) in enumerate(seeds):
            labels[r, c] = k
        counts = [1] * components
        while any(counts[k] < targets[k] for k in range(components)):
            frontiers = []
            for k in range(components):
                if counts[k] >= targets[k]:
                    continue
                options = []
                for r, c in np.argwhere(labels == k):
                    for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= a < rows and 0 <= b < cols and labels[a, b] < 0:
                            options.append((a, b))
                if options:
                    frontiers.append((k, options))
            if not frontiers:
                break
            weights = np.array([targets[k] - counts[k] for k, _ in frontiers], dtype=float)
            weights /= weights.sum()
            k, options = frontiers[int(rng.choice(len(frontiers), p=weights))]
            a, b = options[int(rng.integers(len(options)))]
            labels[a, b] = k
            counts[k] += 1
        if all(counts[k] == targets[k] for k in range(components)):
            return labels
    return None


def _pad_start(rng, band_start, band_size, pad_size, lane):
    local_lane = lane - band_start
    lo = max(0, local_lane - pad_size + 2)
    hi = min(band_size - pad_size, local_lane - 1)
    return band_start + int(rng.integers(lo, hi + 1))


def _pair_geometry(rng, height, width, regime, force_multi=False):
    maximum_rows = min(5, (height + 1) // 4)
    maximum_cols = min(5, (width + 1) // 4)
    if maximum_rows < 2 or maximum_cols < 2:
        return None
    rows = int(rng.integers(2, maximum_rows + 1))
    cols = int(rng.integers(2, maximum_cols + 1))
    if height >= 22 and width >= 22 and rng.random() < (0.7 if regime == "target_test" else 0.45):
        rows = max(rows, 4)
        cols = max(cols, 4)
    slots = rows * cols
    high = min(5, slots // 2)
    components = int(rng.integers(2 if force_multi else 1, high + 1))
    minimum = 2 * components
    maximum = min(slots, 10 * components)
    density = rng.uniform(0.45, 0.85) if regime == "source_train" else rng.uniform(0.55, 0.96)
    total = max(minimum, min(maximum, int(round(slots * density))))
    big = (height >= 21 and width >= 23 and rows <= 4 and cols <= 4
           and rng.random() < (0.37 if regime == "target_test" else 0.19))
    axis_y = _axis(rng, height, rows, large=4 if big else False,
                   long_gap=rng.random() < 0.4)
    axis_x = _axis(rng, width, cols, large=6 if big else False,
                   long_gap=rng.random() < 0.4)
    if axis_y is None or axis_x is None:
        return None
    ysizes, ystarts, ylanes, big_r, ygaps = axis_y
    xsizes, xstarts, xlanes, big_c, xgaps = axis_x
    required = (big_r, big_c) if big else None
    labels = _slot_labels(rng, rows, cols, components, total, required)
    if labels is None:
        return None
    masks = np.zeros((components, height, width), dtype=bool)
    pads = {}
    for r, c in np.argwhere(labels >= 0):
        r, c = int(r), int(c)
        k = int(labels[r, c])
        is_big = big and (r, c) == required
        ph = ysizes[r] if is_big else int(rng.integers(3, ysizes[r] + 1))
        pw = xsizes[c] if is_big else int(rng.integers(3, xsizes[c] + 1))
        y = _pad_start(rng, ystarts[r], ysizes[r], ph, ylanes[r])
        x = _pad_start(rng, xstarts[c], xsizes[c], pw, xlanes[c])
        pads[(r, c)] = (y, x, ph, pw)
        masks[k, y:y + ph, x:x + pw] = True
    for (r, c), (y, x, ph, pw) in pads.items():
        k = int(labels[r, c])
        if (r, c + 1) in pads and labels[r, c + 1] == k:
            other_x = pads[(r, c + 1)][1]
            masks[k, ylanes[r], x + pw:other_x] = True
        if (r + 1, c) in pads and labels[r + 1, c] == k:
            other_y = pads[(r + 1, c)][0]
            masks[k, y + ph:other_y, xlanes[c]] = True
    if np.any(np.sum(masks, axis=0) > 1):
        return None
    if len(_components(np.any(masks, axis=0))) != components:
        return None
    return masks, labels, pads, big, (ygaps, xgaps)


def _make_pair(rng, shape, regime, stable_roles=None, force_multi=False,
               familiar=None):
    height, width = shape
    for _ in range(90):
        geometry = _pair_geometry(rng, height, width, regime, force_multi)
        if geometry is None:
            continue
        masks, labels, pads, big, gaps = geometry
        if stable_roles is None:
            bg, structure = [int(x) for x in rng.choice(10, size=2, replace=False)]
        else:
            bg, structure = stable_roles
        available = [c for c in range(10) if c not in (bg, structure)]
        marker_colors = []
        for k in range(len(masks)):
            choices = available
            if familiar and regime == "target_test" and rng.random() < 0.55:
                known = [c for c in available if c in familiar]
                if known:
                    choices = known
            if marker_colors and rng.random() < 0.17:
                color = int(rng.choice(marker_colors))
            else:
                color = int(rng.choice(choices))
            marker_colors.append(color)
        inp = np.full((height, width), bg, dtype=np.int8)
        for mask in masks:
            inp[mask] = structure
        for k, color in enumerate(marker_colors):
            slots = [(int(r), int(c)) for r, c in np.argwhere(labels == k)]
            r, c = slots[int(rng.integers(len(slots)))]
            y, x, ph, pw = pads[(r, c)]
            inner = _interior(masks[k])
            choices = np.argwhere(inner[y + 1:y + ph - 1, x + 1:x + pw - 1])
            if len(choices) == 0:
                break
            dy, dx = choices[int(rng.integers(len(choices)))]
            inp[y + 1 + int(dy), x + 1 + int(dx)] = color
        else:
            try:
                out = _predict(inp)
            except AssertionError:
                continue
            if np.array_equal(inp, out):
                continue
            return {"input": inp, "output": out}
    raise RuntimeError(f"unable to construct {regime} pair of shape {shape}")


def _shape(rng, regime, index):
    if regime == "source_train" and index == 0:
        h, w = int(rng.integers(22, 31)), int(rng.integers(22, 31))
    elif regime == "source_train" and index == 1:
        h, w = int(rng.integers(15, 25)), int(rng.integers(15, 25))
    elif regime == "target_test" and rng.random() < 0.62:
        h, w = int(rng.integers(24, 31)), int(rng.integers(24, 31))
    else:
        h, w = int(rng.integers(12, 31)), int(rng.integers(12, 31))
    return h, w


def _key(grid):
    return grid.shape, grid.tobytes()


def _pair_hash(pair):
    data = b"".join(str(pair[k].shape).encode() + pair[k].tobytes()
                    for k in ("input", "output"))
    return hashlib.sha256(data).hexdigest()


def _pair_features(inp):
    """Measure visible geometry without using construction parameters."""
    bg, structure, regions, markers, interiors = _roles_and_components(inp)
    foreground = inp != bg
    boxes = []
    networks = []
    corridor_lengths = []
    for cells, inner in zip(regions, interiors):
        cy, cx = zip(*cells)
        boxes.append((min(cy), max(cy), min(cx), max(cx)))
        mask = np.zeros_like(foreground)
        mask[cy, cx] = True
        pad_boxes = []
        sizes = []
        for core in _components(inner):
            yy, xx = zip(*core)
            y0, y1, x0, x1 = min(yy) - 1, max(yy) + 1, min(xx) - 1, max(xx) + 1
            assert len(core) == (max(yy) - min(yy) + 1) * (max(xx) - min(xx) + 1)
            pad_boxes.append((y0, y1, x0, x1))
            sizes.append((y1 - y0 + 1, x1 - x0 + 1))
        pad_ids = np.full(inp.shape, -1, dtype=np.int16)
        for j, (y0, y1, x0, x1) in enumerate(pad_boxes):
            pad_ids[y0:y1 + 1, x0:x1 + 1] = j
        degrees = []
        for j, (y0, y1, x0, x1) in enumerate(pad_boxes):
            directions = 0
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if dy:
                    starts = [(y0 - 1 if dy < 0 else y1 + 1, x)
                              for x in range(x0 + 1, x1)]
                else:
                    starts = [(y, x0 - 1 if dx < 0 else x1 + 1)
                              for y in range(y0 + 1, y1)]
                connected = False
                for y, x in starts:
                    if not (0 <= y < inp.shape[0] and 0 <= x < inp.shape[1] and mask[y, x]):
                        continue
                    length = 0
                    while (0 <= y < inp.shape[0] and 0 <= x < inp.shape[1]
                           and mask[y, x] and pad_ids[y, x] < 0):
                        length += 1
                        y += dy
                        x += dx
                    if (0 <= y < inp.shape[0] and 0 <= x < inp.shape[1]
                            and pad_ids[y, x] >= 0 and pad_ids[y, x] != j):
                        connected = True
                        if dy > 0 or dx > 0:
                            corridor_lengths.append(length)
                directions += int(connected)
            degrees.append(directions)
        networks.append({"sizes": sizes, "degrees": degrees})
    overlap = any(max(a[0], b[0]) <= min(a[1], b[1]) and
                  max(a[2], b[2]) <= min(a[3], b[3])
                  for i, a in enumerate(boxes) for b in boxes[i + 1:])

    def zone_leaves(y0, y1, x0, x1):
        if not foreground[y0:y1, x0:x1].any():
            return []
        for y in range(y0, y1):
            if not foreground[y, x0:x1].any():
                return zone_leaves(y0, y, x0, x1) + zone_leaves(y + 1, y1, x0, x1)
        for x in range(x0, x1):
            if not foreground[y0:y1, x].any():
                return zone_leaves(y0, y1, x0, x) + zone_leaves(y0, y1, x + 1, x1)
        return [(y0, y1, x0, x1)]

    marker_cells = [next((y, x) for y, x in cells if inp[y, x] != structure)
                    for cells in regions]
    zone_shortcut = all(sum(y0 <= y < y1 and x0 <= x < x1
                            for y, x in marker_cells) <= 1
                        for y0, y1, x0, x1 in zone_leaves(0, inp.shape[0], 0, inp.shape[1]))
    return {"background": bg, "structure": structure, "markers": markers,
            "networks": networks, "overlap": overlap,
            "zone_shortcut": zone_shortcut, "corridor_lengths": corridor_lengths,
            "border": bool(np.any(foreground[0]) or np.any(foreground[-1]) or
                           np.any(foreground[:, 0]) or np.any(foreground[:, -1]))}


def _independent_prediction(inp):
    """Self-test path: explicit neighborhood and flood fill from input only."""
    bg, structure, _, _, _ = _roles_and_components(inp)
    h, w = inp.shape
    visited = set()
    expected = inp.copy()
    for y in range(h):
        for x in range(w):
            if inp[y, x] == bg or (y, x) in visited:
                continue
            todo = [(y, x)]
            visited.add((y, x))
            cells = set()
            while todo:
                a, b = todo.pop()
                cells.add((a, b))
                for c, d in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if (0 <= c < h and 0 <= d < w and inp[c, d] != bg
                            and (c, d) not in visited):
                        visited.add((c, d))
                        todo.append((c, d))
            seed = [(a, b) for a, b in cells if inp[a, b] != structure]
            assert len(seed) == 1
            color = inp[seed[0]]
            for a, b in cells:
                if all((a + dy, b + dx) in cells
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                    expected[a, b] = color
    return expected


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nr, nt = _resolve_counts(rng, seed, num_train, num_test)
    # The transformation has no fixed literal colors. Some puzzles retain a
    # palette; others demonstrate that both role colors can change by pair.
    stable = rng.random() < 0.25
    roles = tuple(int(v) for v in rng.choice(10, size=2, replace=False)) if stable else None
    puzzle = {"train": [], "test": []}
    seen = set()
    familiar = set()
    for regime, amount, section in (("source_train", nr, "train"),
                                     ("target_test", nt, "test")):
        for i in range(amount):
            for _ in range(20):
                shape = _shape(rng, regime, i)
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                try:
                    pair = _make_pair(child, shape, regime, roles,
                                      force_multi=(section == "train" and i == 0),
                                      familiar=familiar)
                except RuntimeError:
                    continue
                if _key(pair["input"]) in seen or _pair_hash(pair) in _SOURCE_PAIR_HASHES:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[section].append(pair)
                _, _, _, markers, _ = _roles_and_components(pair["input"])
                if section == "train":
                    familiar.update(markers)
                break
            else:
                raise RuntimeError(f"unable to construct unique {section} pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nr, nt = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nr <= 10 and 1 <= nt <= 4 and nr > nt
    seen = set()
    train_markers = set()
    orientations = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            assert _pair_hash(pair) not in _SOURCE_PAIR_HASHES
            bg, structure, regions, markers, interiors = _roles_and_components(inp)
            assert np.array_equal(out, _predict(inp)) and not np.array_equal(inp, out)
            assert len(regions) <= 5
            assert all(len(_components(inner)) >= 2 for inner in interiors)
            for inner in interiors:
                for core in _components(inner):
                    yy, xx = zip(*core)
                    assert len(core) == (max(yy) - min(yy) + 1) * (max(xx) - min(xx) + 1)
            for cells in regions:
                for y, x in cells:
                    if 0 < x < inp.shape[1] - 1 and inp[y, x - 1] != bg and inp[y, x + 1] != bg:
                        if (y == 0 or inp[y - 1, x] == bg) and (y == inp.shape[0] - 1 or inp[y + 1, x] == bg):
                            orientations.add("horizontal")
                    if 0 < y < inp.shape[0] - 1 and inp[y - 1, x] != bg and inp[y + 1, x] != bg:
                        if (x == 0 or inp[y, x - 1] == bg) and (x == inp.shape[1] - 1 or inp[y, x + 1] == bg):
                            orientations.add("vertical")
            if section == "train":
                train_markers.update(markers)
    assert len(train_markers) >= 2


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    for nr, nt in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None), (6, 3)):
        p = generate_puzzle(900 + (nr or 0) * 10 + (nt or 0), nr, nt)
        assert nr is None or len(p["train"]) == nr
        assert nt is None or len(p["test"]) == nt
    for nr, nt in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(1, nr, nt)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    assert _equal(generate_puzzle(417), generate_puzzle(417))
    assert not _equal(generate_puzzle(417), generate_puzzle(418))
    train_counts, test_counts, shapes, roles, layouts = set(), set(), set(), set(), set()
    pad_counts, border, overlap, long_corridors = set(), 0, 0, 0
    branching = four_way = multi_junction = mixed_sizes = large_train = large_test = 0
    resistant = same_seed = familiar_test = novel_test = palette_changes = 0
    train_pad_totals, test_pad_totals = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        puzzle_roles = set()
        known = set()
        for section in ("train", "test"):
            for pair in p[section]:
                inp = pair["input"]
                bg, st, regions, markers, interiors = _roles_and_components(inp)
                assert np.array_equal(_independent_prediction(inp), pair["output"])
                features = _pair_features(inp)
                puzzle_roles.add((bg, st))
                roles.add((bg, st))
                shapes.add(inp.shape)
                occupancy = inp != bg
                layouts.add((inp.shape, occupancy.tobytes()))
                border += int(features["border"])
                overlap += int(features["overlap"])
                resistant += int(not features["zone_shortcut"])
                same_seed += int(len(set(markers)) < len(markers))
                long_corridors += sum(length >= 4 for length in features["corridor_lengths"])
                pad_total = sum(len(net["sizes"]) for net in features["networks"])
                (train_pad_totals if section == "train" else test_pad_totals).append(pad_total)
                large = any((h >= 7 and w >= 9) or (w >= 7 and h >= 9)
                            for net in features["networks"] for h, w in net["sizes"])
                if section == "train":
                    large_train += int(large)
                    known.update(markers)
                else:
                    large_test += int(large)
                    familiar_test += int(bool(set(markers) & known))
                    novel_test += int(bool(set(markers) - known))
                for net in features["networks"]:
                    pad_counts.add(len(net["sizes"]))
                    branching += int(any(d >= 3 for d in net["degrees"]))
                    four_way += int(any(d == 4 for d in net["degrees"]))
                    multi_junction += int(sum(d >= 3 for d in net["degrees"]) >= 2)
                    mixed_sizes += int(len(set(net["sizes"])) > 1)
        palette_changes += int(len(puzzle_roles) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(roles) >= 20 and len(layouts) >= 100
    assert border >= 20 and overlap >= 5
    assert resistant >= 20 and palette_changes >= 10 and same_seed >= 5
    assert branching >= 20 and four_way >= 2 and multi_junction >= 2
    assert mixed_sizes >= 50 and large_train >= 2 and large_test >= 2
    assert long_corridors >= 20 and max(pad_counts) >= 8
    assert familiar_test >= 20 and novel_test >= 3
    assert min(train_pad_totals) <= 4 and max(train_pad_totals) >= 12
    assert min(test_pad_totals) <= 4 and max(test_pad_totals) >= 12
