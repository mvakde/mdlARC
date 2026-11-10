"""Generate local plurality fills inside closed, one-cell ARC contours."""

import zlib

import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SOURCE_PAIRS = (
    ((19, 18), 1327146342, 3364117912,
     "789c6d4ed10e0031049323e2ffbff886f456397d98aa9689249e8400e848a957b3babbd02c9533e854915246ae533717b4993cf33efea3b22040ece0cad62d2b288145962c56cf543e8fae9bffd7092fcd970132",
     "789ca58ed10e002004450d33ffffc5a1554aeba5f376cf5c0038e8c020121125431d3322729ad93acc98eaaddc5b9baf06f77f1eb0b12756cd46837f5336d7eb89066695023c"),
    ((21, 21), 2684943013, 3433939807,
     "789cad8f410ac0300804d748c0ffbfb86aa4b0d6420f9d90cbb8b209f0c63a34e97763b51cac46943b23929eb354ea5032e5d4adb1a46094e38394a08aaa2a85092f49fc074f89416e3f9f92d9dde4fd829fb800d18500c2",
     "789cadcd4b0a00200845513384f6bfe27ce8c44f04d11dc5d190e8145b0917e2b4d7a0117788e7d42a76b7679c58771cc80ff929051119f113e83ba28c3ea8f8d806ebe001c6"),
    ((16, 14), 1568962941, 2918673671,
     "789c7d4ed90a0030089209b1ffffe2d181480f93283ca880050e8601a7473912bbdd2c31f394e488517b08cfd212b634b362013db3eed8933f3c551e00b1",
     "789c63604003cc5000e5b1810133540699c7c6869b0757c98c6c0a1b161e33331e632046401dc38c3001ee4c74c7630200101b0235"),
    ((26, 24), 309520901, 2964780903,
     "789c75500b0ec52008ab9f28f73ff1030b6cf8b2264e2ca5ab029fe884560d180756a147d7b6911fe51720de576a257fc66c370bd7fbbcf6ba1b087953afd00b52ff64222185470a757c94f419303687df2af956bac3f3d4195c9c18c8bb97f35c6f4ba1564e0f555a0b6611ccc85160c719c526783c9e93fc91ee3fffad2fd59267f85b4f92f7dad7ab04f2aff9269f780c4cf6030b6402df",
     "789c9d905b0ec0200804a592c8fd4f5cca62d5e223e9fc39882ea4b4e4023864039e1ebc9245e9bd56d65e9fa85efb3b2ff27aa28df744d17bcace0fe99b1fc80814bcb79cbd3ddb16d43c33cbd8880813cfce1f5f40f476f5e411e8eb6d229fab4c3690eabf8bda961b1ff504fd"),
)
VOTE_PATTERNS = (
    (2, 1), (3, 1), (3, 2), (4, 1), (4, 2),
    (3, 2, 1), (3, 2, 2), (4, 2, 1), (4, 3, 1),
    (4, 2, 1, 1), (5, 3, 2),
)


def _check(test, message):
    if not test:
        raise AssertionError(message)


def _key(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _source_copy(pair):
    a, b = pair["input"], pair["output"]
    a_bytes = a.astype(np.uint8, copy=False).tobytes()
    b_bytes = b.astype(np.uint8, copy=False).tobytes()
    signature = a.shape, zlib.crc32(a_bytes), zlib.crc32(b_bytes)
    for shape, input_crc, output_crc, compressed_input, compressed_output in SOURCE_PAIRS:
        if signature == (shape, input_crc, output_crc):
            return (a_bytes == zlib.decompress(bytes.fromhex(compressed_input)) and
                    b_bytes == zlib.decompress(bytes.fromhex(compressed_output)))
    return False


def _neighbors(r, c, h, w):
    for dr, dc in NEIGHBORS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack, coords = [(r, c)], []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(coords)
    return result


def _interior(barrier):
    wall = np.pad(barrier, 1)
    h, w = wall.shape
    outside = np.zeros_like(wall)
    outside[0, 0] = True
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, h, w):
            if not wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return (~wall & ~outside)[1:-1, 1:-1]


def _degrees(mask, coords):
    h, w = mask.shape
    return [sum(mask[nr, nc] for nr, nc in _neighbors(r, c, h, w))
            for r, c in coords]


def _analyze(grid):
    h, w = grid.shape
    background = int(np.bincount(grid.ravel().astype(int), minlength=10).argmax())
    _check(int(np.count_nonzero(grid == background)) > grid.size // 2,
           "input needs a clear majority background")
    structures, marker_mask = [], np.zeros((h, w), bool)
    for color in range(10):
        if color == background:
            continue
        for coords in _components(grid == color):
            if len(coords) == 1:
                marker_mask[coords[0]] = True
                continue
            _check(len(coords) >= 8, "colored components must be paths or single markers")
            mask = np.zeros((h, w), bool)
            for pos in coords:
                mask[pos] = True
            interior = _interior(mask)
            degrees = _degrees(mask, coords)
            if np.any(interior):
                _check(all(d == 2 for d in degrees), "closed contour has a branch")
                _check(len(_components(interior)) == 1, "closed contour has multiple rooms")
                closed = True
            else:
                _check(all(d in (1, 2) for d in degrees) and degrees.count(1) == 2,
                       "open contour is not a single path")
                closed = False
            rows, cols = zip(*coords)
            bbox = min(rows), max(rows), min(cols), max(cols)
            structures.append(dict(color=color, mask=mask, interior=interior,
                                   closed=closed, bbox=bbox, size=len(coords)))
    closed = [x for x in structures if x["closed"]]
    opened = [x for x in structures if not x["closed"]]
    _check(closed, "at least one closed contour is required")
    _check(len(structures) <= 4, "too many contours")
    all_structure = np.zeros((h, w), bool)
    for x in structures:
        _check(not np.any(all_structure & x["mask"]), "overlapping contours")
        all_structure |= x["mask"]
    owned = np.zeros((h, w), bool)
    for x in closed:
        inside = x["interior"]
        _check(not np.any(inside & (all_structure | owned)), "nested contours")
        owned |= inside
        values = grid[inside & marker_mask]
        counts = {int(color): int(np.count_nonzero(values == color))
                  for color in np.unique(values)}
        _check(len(counts) >= 2, "closed region needs competing colors")
        maximum = max(counts.values())
        winners = [color for color, count in counts.items() if count == maximum]
        _check(len(winners) == 1, "closed region lacks unique plurality")
        _check(winners[0] != x["color"], "fill must remain visible beside boundary")
        x["votes"], x["winner"] = counts, winners[0]
    _check(not np.any(owned & (grid != background) & ~marker_mask),
           "interior contains non-marker objects")
    for x in opened:
        r0, r1, c0, c1 = x["bbox"]
        x["apparent_markers"] = int(np.count_nonzero(marker_mask[r0 + 1:r1, c0 + 1:c1]))
    box_union = np.zeros((h, w), bool)
    for x in structures:
        r0, r1, c0, c1 = x["bbox"]
        box_union[r0:r1 + 1, c0:c1 + 1] = True
    contacts = 0
    same_local = 0
    for r, c in np.argwhere(marker_mask):
        near_structure = any(all_structure[nr, nc] for nr, nc in _neighbors(int(r), int(c), h, w))
        contacts += int(near_structure)
        for x in closed:
            if x["interior"][r, c] and grid[r, c] == x["color"]:
                same_local += 1
    return dict(background=background, structures=structures, closed=closed,
                open=opened, marker_mask=marker_mask, structure_mask=all_structure,
                contacts=contacts, same_local=same_local,
                exterior=int(np.count_nonzero(marker_mask & ~box_union)),
                palette=tuple(int(x) for x in np.unique(grid)))


def _solve(grid, analysis=None):
    if analysis is None:
        analysis = _analyze(grid)
    output = np.full(grid.shape, analysis["background"], np.int8)
    output[analysis["structure_mask"]] = grid[analysis["structure_mask"]]
    for x in analysis["closed"]:
        output[x["interior"]] = x["winner"]
    return output


def _line(mask, a, b):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        mask[r0, min(c0, c1):max(c0, c1) + 1] = True
    elif c0 == c1:
        mask[min(r0, r1):max(r0, r1) + 1, c0] = True
    else:
        raise AssertionError("non-orthogonal contour segment")


def _cycle(height, width, rng, complexity):
    pad = (int(rng.integers(1, 3)) if complexity > 0 and min(height, width) >= 11
           else int(complexity > 0 and min(height, width) >= 9))
    top, left = pad, pad
    bottom, right = height - 1 - pad, width - 1 - pad
    corners = ((top, left), (top, right), (bottom, right), (bottom, left))
    points = []
    sides = rng.permutation(4).tolist()
    varied = set(int(x) for x in sides[:complexity])
    for side in range(4):
        start, end = corners[side], corners[(side + 1) % 4]
        length = abs(end[0] - start[0]) + abs(end[1] - start[1])
        along = ((end[0] - start[0]) // length, (end[1] - start[1]) // length)
        inward = ((1, 0), (0, -1), (-1, 0), (0, 1))[side]
        intervals = []
        if side in varied and length >= 6:
            count = 2 if length >= 12 and rng.random() < .35 else 1
            if count == 2:
                starts = (2, length // 2 + 1)
                for a in starts:
                    b = min(length - 2, a + int(rng.integers(2, 4)))
                    if b - a >= 2:
                        intervals.append((a, b))
            else:
                a = int(rng.integers(2, length - 2))
                b = int(rng.integers(a + 2, length)) if a + 2 < length else length - 1
                b = min(b, length - 2)
                if b - a >= 2:
                    intervals.append((a, b))
        points.append(start)
        for a, b in intervals:
            max_depth = min(3, max(1, min(height, width) // 3 - 1))
            depth = int(rng.integers(1, max_depth + 1))
            if pad and rng.random() < .35:
                depth = -int(rng.integers(1, pad + 1))
            pa = (start[0] + along[0] * a, start[1] + along[1] * a)
            pb = (start[0] + along[0] * b, start[1] + along[1] * b)
            offset = (inward[0] * depth, inward[1] * depth)
            points.extend((pa, (pa[0] + offset[0], pa[1] + offset[1]),
                           (pb[0] + offset[0], pb[1] + offset[1]), pb))
        points.append(end)
    mask = np.zeros((height, width), bool)
    for a, b in zip(points, points[1:] + points[:1]):
        _line(mask, a, b)
    rows, cols = np.where(mask)
    mask = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return mask


def _ordered_cycle(mask):
    coords = _components(mask)[0]
    start = min(coords)
    h, w = mask.shape
    neighbors = [p for p in _neighbors(*start, h, w) if mask[p]]
    if len(neighbors) != 2:
        return []
    ordered, previous, current = [start], start, neighbors[0]
    while current != start and len(ordered) <= len(coords):
        ordered.append(current)
        next_cells = [p for p in _neighbors(*current, h, w)
                      if mask[p] and p != previous]
        if len(next_cells) != 1:
            return []
        previous, current = current, next_cells[0]
    return ordered if len(ordered) == len(coords) else []


def _shape(slot_h, slot_w, opened, marker_count, rng, complexity):
    minimum = 5 if opened else 6
    for _ in range(80):
        h = int(rng.integers(minimum, slot_h + 1))
        w = int(rng.integers(minimum, slot_w + 1))
        cycle = _cycle(h, w, rng, complexity)
        coords = _components(cycle)
        if len(coords) != 1 or any(d != 2 for d in _degrees(cycle, coords[0])):
            continue
        if len(_components(_interior(cycle))) != 1:
            continue
        if opened:
            ordered = _ordered_cycle(cycle)
            if not ordered:
                continue
            gap = int(rng.integers(1, max(3, min(len(ordered) // 3, max(h, w)) + 1)))
            start = int(rng.integers(0, len(ordered)))
            for k in range(gap):
                cycle[ordered[(start + k) % len(ordered)]] = False
            if np.any(_interior(cycle)) or len(_components(cycle)) != 1:
                continue
            remaining = _components(cycle)[0]
            deg = _degrees(cycle, remaining)
            if len(remaining) < 8 or deg.count(1) != 2 or any(d not in (1, 2) for d in deg):
                continue
            rr, cc = np.where(cycle)
            apparent = np.zeros_like(cycle)
            apparent[rr.min() + 1:rr.max(), cc.min() + 1:cc.max()] = True
            allowed = apparent & ~cycle
        else:
            allowed = _interior(cycle)
        if max(int(np.count_nonzero(allowed & (np.indices(allowed.shape).sum(axis=0) % 2 == p)))
               for p in (0, 1)) >= marker_count:
            return cycle, allowed
    raise RuntimeError("could not form contour with enough marker space")


def _slots(height, width, count, rng):
    slots = [(0, height, 0, width)]
    while len(slots) < count:
        index = max(range(len(slots)), key=lambda i: (slots[i][1] - slots[i][0]) * (slots[i][3] - slots[i][2]))
        r0, r1, c0, c1 = slots.pop(index)
        can_h = r1 - r0 >= 14
        can_w = c1 - c0 >= 14
        _check(can_h or can_w, "canvas cannot fit its contour count")
        horizontal = bool(can_h and (not can_w or (r1 - r0) > (c1 - c0) * float(rng.uniform(.75, 1.25))))
        gap = int(rng.integers(1, 3))
        if horizontal:
            minimum = 7 if r1 - r0 - gap >= 14 else 6
            cut = int(rng.integers(r0 + minimum, r1 - gap - minimum + 1))
            parts = [(r0, cut, c0, c1), (cut + gap, r1, c0, c1)]
        else:
            minimum = 7 if c1 - c0 - gap >= 14 else 6
            cut = int(rng.integers(c0 + minimum, c1 - gap - minimum + 1))
            parts = [(r0, r1, c0, cut), (r0, r1, cut + gap, c1)]
        slots.extend(parts)
    return [slots[int(i)] for i in rng.permutation(len(slots))]


def _dimensions(count, regime, rng):
    low = (11, 16, 22, 26)[count - 1]
    if regime == "target_test" and rng.random() < .38:
        low = min(30, low + 2)
    h = int(rng.integers(low, 31 if count >= 3 else min(30, low + 13) + 1))
    w = int(rng.integers(low, 31 if count >= 3 else min(30, low + 13) + 1))
    return h, w


def _votes(rng, forced=None):
    pattern = forced if forced is not None else VOTE_PATTERNS[int(rng.integers(len(VOTE_PATTERNS)))]
    return tuple(int(x) for x in pattern)


def _place_markers(grid, allowed, colors, contour_color, background, rng):
    h, w = grid.shape
    coords = np.argwhere(allowed & (grid == background))
    parity = (coords[:, 0] + coords[:, 1]) % 2
    groups = [coords[parity == p] for p in (0, 1)]
    groups = [g for g in groups if len(g) >= len(colors)]
    if not groups:
        return False
    coords = groups[int(rng.integers(len(groups)))]
    marker_colors = [int(x) for x in rng.permutation(colors)]
    used = set()
    for color in marker_colors:
        placed = False
        order = [tuple(map(int, p)) for p in coords]
        rng.shuffle(order)
        near = [p for p in order if any(grid[nr, nc] == contour_color
                                          for nr, nc in _neighbors(*p, h, w))]
        far = [p for p in order if p not in near]
        candidates = (near + far) if rng.random() < .25 else (far + near)
        for r, c in candidates:
            r, c = int(r), int(c)
            if (r, c) in used:
                continue
            if any(grid[nr, nc] == color for nr, nc in _neighbors(r, c, h, w)):
                continue
            grid[r, c] = color
            used.add((r, c))
            placed = True
            break
        if not placed:
            return False
    return True


def _make_pair(rng, regime, anchor, anchor_winner, open_required):
    for _ in range(100):
        closed_count = int(rng.choice([1, 2, 3], p=[.38, .47, .15] if regime == "source_train" else [.22, .43, .35]))
        open_count = int(rng.choice([0, 1, 2], p=[.48, .42, .10] if regime == "source_train" else [.37, .43, .20]))
        if open_required:
            open_count = max(open_count, 1)
        open_count = min(open_count, 4 - closed_count)
        count = closed_count + open_count
        height, width = _dimensions(count, regime, rng)
        slots = _slots(height, width, count, rng)
        forbidden_bg = {anchor} if anchor_winner is None else {anchor, anchor_winner}
        background = int(rng.choice([c for c in range(10) if c not in forbidden_bg]))
        grid = np.full((height, width), background, np.int8)
        records = []
        object_boxes = []
        random_colors = [c for c in range(10) if c != background]
        contour_colors = [int(anchor) if anchor_winner is not None else int(rng.choice(random_colors))]
        contour_colors.extend(int(rng.choice(random_colors)) for _ in range(closed_count - 1))
        contour_colors.extend(int(rng.choice(random_colors)) for _ in range(open_count))
        shared_winner = None
        if closed_count > 1 and rng.random() < .38:
            shared_winner = int(rng.choice([c for c in random_colors if c not in contour_colors[:closed_count]])) if len(set(contour_colors[:closed_count])) < 8 else None
        failed = False
        for i, slot in enumerate(slots):
            opened = i >= closed_count
            color = contour_colors[i]
            if opened:
                vote_colors = [int(x) for x in rng.choice(random_colors, size=int(rng.integers(1, 4)), replace=True)]
            else:
                pattern = _votes(rng)
                winner = (int(anchor_winner) if i == 0 and anchor_winner is not None
                          else shared_winner if shared_winner is not None and shared_winner != color
                          else int(rng.choice([c for c in random_colors if c != color])))
                candidates = [c for c in random_colors if c != winner]
                if color in candidates and rng.random() < .30:
                    losers = [color]
                    candidates.remove(color)
                else:
                    losers = []
                rng.shuffle(candidates)
                losers += candidates[:len(pattern) - 1 - len(losers)]
                vote_colors = [winner] * pattern[0]
                for loser, votes in zip(losers, pattern[1:]):
                    vote_colors.extend([loser] * votes)
            r0, r1, c0, c1 = slot
            complexity = int(rng.choice([0, 1, 2, 3], p=[.23, .30, .30, .17]))
            try:
                local, allowed = _shape(r1 - r0, c1 - c0, opened, len(vote_colors), rng, complexity)
            except RuntimeError:
                failed = True
                break
            lh, lw = local.shape
            row = int(rng.integers(r0, r1 - lh + 1))
            col = int(rng.integers(c0, c1 - lw + 1))
            if rng.random() < .23 and r0 == 0:
                row = 0
            if rng.random() < .23 and c0 == 0:
                col = 0
            region = grid[row:row + lh, col:col + lw]
            region[local] = color
            object_boxes.append((row, row + lh, col, col + lw))
            global_allowed = np.zeros_like(grid, bool)
            global_allowed[row:row + lh, col:col + lw] = allowed
            records.append((global_allowed, vote_colors, color, opened))
        if failed:
            continue
        for allowed, colors, color, opened in records:
            if not _place_markers(grid, allowed, colors, color, background, rng):
                failed = True
                break
        if failed:
            continue
        exterior = int(rng.choice([0, 1, 2, 3, 4, 5], p=[.21, .19, .20, .17, .14, .09]))
        outside = np.ones_like(grid, bool)
        for r0, r1, c0, c1 in object_boxes:
            outside[r0:r1, c0:c1] = False
        if not outside.any():
            exterior = 0
        extra_colors = [int(rng.choice(random_colors)) for _ in range(exterior)]
        if not _place_markers(grid, outside, extra_colors, -1, background, rng):
            continue
        try:
            analysis = _analyze(grid)
        except AssertionError:
            continue
        if len(analysis["closed"]) != closed_count or len(analysis["open"]) != open_count:
            continue
        output = _solve(grid, analysis)
        pair = {"input": grid, "output": output}
        if not _source_copy(pair):
            return pair
    raise RuntimeError("unable to generate a valid contour pair")


def _count_arg(value, name, low, high):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _counts(rng, num_train, num_test):
    _count_arg(num_train, "num_train", 2, 10)
    _count_arg(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    anchor = int(rng.integers(0, 10))
    winners = [int(x) for x in rng.choice([c for c in range(10) if c != anchor], size=2, replace=False)]
    open_demo = int(rng.integers(0, 2))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for i in range(count):
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                  anchor, winners[i] if split == "train" and i < 2 else None,
                                  split == "train" and i == open_demo)
                key = _key(pair["input"])
                if key not in used:
                    puzzle[split].append(pair)
                    used.add(key)
                    break
            else:
                raise RuntimeError("could not find a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D NumPy array")
    _check(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _check(all(1 <= n <= 30 for n in grid.shape), "grid dimensions must be 1..30")
    _check(np.all((grid >= 0) & (grid <= 9)), "ARC colors must be 0..9")


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pairs must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_analyses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, b = pair["input"], pair["output"]
            _validate_grid(a)
            _validate_grid(b)
            _check(a.shape == b.shape and not np.array_equal(a, b), "output must visibly transform input")
            _check(not _source_copy(pair), "pair duplicates the source task")
            key, full = _key(a), (_key(a), _key(b))
            _check(key not in seen_inputs and full not in seen_pairs, "duplicate pair")
            seen_inputs.add(key)
            seen_pairs.add(full)
            analysis = _analyze(a)
            _check(np.array_equal(b, _solve(a, analysis)), "incorrect plurality fill or marker cleanup")
            _check(1 <= len(analysis["closed"]) and len(analysis["structures"]) <= 4,
                   "invalid contour count")
            if split == "train":
                train_analyses.append(analysis)
    _check(any(a["open"] for a in train_analyses), "training needs an open decoy")
    # A repeated contour color with two distinct winners refutes a color lookup.
    contour_to_winners = {}
    for a in train_analyses:
        for x in a["closed"]:
            contour_to_winners.setdefault(x["color"], set()).add(x["winner"])
    _check(any(len(v) >= 2 for v in contour_to_winners.values()),
           "training must show that contour color does not determine fill")


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(4200, 4242)]
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "too few train counts")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "too few test counts")
    shapes, structures, palettes, votes, contacts, same_local = set(), set(), set(), set(), 0, 0
    open_counts, exterior_counts, backgrounds = set(), set(), set()
    edge_contours = winding = small_open = wide_gap = shared_winners = 0
    zero_exterior = many_exterior = one_open_marker = two_open = combined = 0
    new_test_contour = new_test_winner = test_not_larger = 0
    within_background = within_palette = within_votes = within_objects = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_analyses = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                a = _analyze(grid)
                puzzle_analyses[split].append(a)
                shapes.add(grid.shape)
                palettes.add(a["palette"])
                backgrounds.add(a["background"])
                open_counts.add(len(a["open"]))
                exterior_counts.add(a["exterior"])
                contacts += a["contacts"]
                same_local += a["same_local"]
                zero_exterior += a["exterior"] == 0
                many_exterior += a["exterior"] >= 4
                two_open += len(a["open"]) == 2
                combined += bool(a["open"]) and a["contacts"] > 0 and a["same_local"] > 0
                winners = [x["winner"] for x in a["closed"]]
                shared_winners += len(winners) > len(set(winners))
                for x in a["structures"]:
                    r0, r1, c0, c1 = x["bbox"]
                    edge_contours += (r0 == 0 or c0 == 0 or
                                      r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1)
                    structures.add((x["closed"], (r1 - r0 + 1, c1 - c0 + 1),
                                    x["mask"][r0:r1 + 1, c0:c1 + 1].tobytes()))
                    coords = _components(x["mask"])[0]
                    turns = 0
                    for r, c in coords:
                        neighbors = [p for p in _neighbors(r, c, *grid.shape) if x["mask"][p]]
                        turns += (len(neighbors) == 2 and neighbors[0][0] != neighbors[1][0]
                                  and neighbors[0][1] != neighbors[1][1])
                    winding += turns >= 8
                for x in a["closed"]:
                    votes.add(tuple(sorted(x["votes"].values(), reverse=True)))
                for x in a["open"]:
                    r0, r1, c0, c1 = x["bbox"]
                    small_open += min(r1 - r0 + 1, c1 - c0 + 1) <= 6
                    one_open_marker += x["apparent_markers"] == 1
                    coords = _components(x["mask"])[0]
                    endpoints = [p for p, d in zip(coords, _degrees(x["mask"], coords)) if d == 1]
                    wide_gap += (len(endpoints) == 2 and
                                 abs(endpoints[0][0] - endpoints[1][0]) +
                                 abs(endpoints[0][1] - endpoints[1][1]) >= 4)
        all_a = puzzle_analyses["train"] + puzzle_analyses["test"]
        within_background += len({a["background"] for a in all_a}) > 1
        within_palette += len({a["palette"] for a in all_a}) > 1
        within_votes += len({tuple(sorted(x["votes"].values(), reverse=True))
                             for a in all_a for x in a["closed"]}) > 1
        within_objects += len({(len(a["closed"]), len(a["open"])) for a in all_a}) > 1
        training_contours = {x["color"] for a in puzzle_analyses["train"] for x in a["structures"]}
        training_winners = {x["winner"] for a in puzzle_analyses["train"] for x in a["closed"]}
        new_test_contour += any(x["color"] not in training_contours
                                for a in puzzle_analyses["test"] for x in a["structures"])
        new_test_winner += any(x["winner"] not in training_winners
                               for a in puzzle_analyses["test"] for x in a["closed"])
        train_h = max(p["input"].shape[0] for p in puzzle["train"])
        train_w = max(p["input"].shape[1] for p in puzzle["train"])
        test_not_larger += any(p["input"].shape[0] <= train_h and
                               p["input"].shape[1] <= train_w for p in puzzle["test"])
    _check(len(shapes) >= 35 and len(structures) >= 50, "insufficient geometric diversity")
    _check(len(palettes) >= 30 and len(backgrounds) >= 5, "insufficient color diversity")
    _check(len(votes) >= 6, "insufficient vote diversity")
    _check(len(open_counts) == 3 and 0 in exterior_counts and max(exterior_counts) >= 4,
           "insufficient decoy diversity")
    _check(contacts > 0 and same_local > 0, "missing source-supported marker contacts")
    _check(edge_contours > 0 and winding > 0 and small_open > 0 and wide_gap > 0,
           "missing boundary, winding, or open-path variations")
    _check(shared_winners > 0 and one_open_marker > 0 and two_open > 0 and combined > 0,
           "missing independent color roles or combined decoy cases")
    _check(zero_exterior > 0 and many_exterior > 0,
           "exterior marker counts have collapsed")
    _check(min(within_background, within_palette, within_votes, within_objects) >= 35,
           "too little within-puzzle variation")
    _check(new_test_contour > 0 and new_test_winner > 0 and test_not_larger > 0,
           "test regime lacks new color roles or size overlap")
    _check(_equal_puzzles(generate_puzzle(912345, 7, 3), generate_puzzle(912345, 7, 3)),
           "generation is not deterministic")
    _check(not _equal_puzzles(generate_puzzle(123456, 6, 2), generate_puzzle(123457, 6, 2)),
           "different seeds match")
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                      (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(930000 + i, train, test)
        if train is not None:
            _check(len(p["train"]) == train, "requested train count ignored")
        if test is not None:
            _check(len(p["test"]) == test, "requested test count ignored")
    for i, (train, test) in enumerate(((1, None), (11, None), (None, 0),
                                      (None, 5), (2, 2), (3, 4), (True, 1), (3, False))):
        try:
            generate_puzzle(950000 + i, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair count accepted")
