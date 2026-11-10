"""One beacon per chamber: exact inventory meets topological reconstruction.

All inference uses the returned input. There are no imported parent generators,
stored examples, external assets, mutable random state, or disk-writing routines.
"""

from collections import Counter, defaultdict
from math import comb, prod

import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    result = np.asarray(value)
    _require(result.ndim == 2 and all(1 <= n <= 30 for n in result.shape),
             "Grid dimensions must be 1..30")
    _require(np.issubdtype(result.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((0 <= result) & (result <= 9))), "Colors must be 0..9")
    return result.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.tobytes()


def _components(mask):
    """Deterministic four-connectivity, also used for exterior segmentation."""
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _DIRECTIONS:
                other = (r + dr, c + dc)
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
        result.append(tuple(sorted(cells)))
    return result


def _normalize(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _model(walls, slots, beacon_cells):
    """Contract each initial floor region; an open plate joins its neighbors.

    Slots are disjoint connected components, hence opening a slot is exactly a
    union of its adjacent floor regions, even for an elbow or branched plate.
    """
    regions = _components(~walls)
    labels = np.full(walls.shape, -1, dtype=int)
    beacon_counts = []
    for index, cells in enumerate(regions):
        for cell in cells:
            labels[cell] = index
        beacon_counts.append(sum(cell in beacon_cells for cell in cells))
    contacts = []
    for cells in slots:
        neighbors = set()
        for r, c in cells:
            for dr, dc in _DIRECTIONS:
                y, x = r + dr, c + dc
                if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                    if labels[y, x] >= 0:
                        neighbors.add(int(labels[y, x]))
        _require(len(neighbors) >= 2, "Every plate must adjoin at least two floor regions")
        contacts.append(tuple(sorted(neighbors)))
    return regions, contacts, tuple(beacon_counts)


def _partition(contacts, node_count, mask):
    parents = list(range(node_count))

    def root(x):
        while parents[x] != x:
            x = parents[x]
        return x

    for index, neighbors in enumerate(contacts):
        if mask & (1 << index):
            for node in neighbors[1:]:
                parents[root(node)] = root(neighbors[0])
    groups = defaultdict(list)
    for node in range(node_count):
        groups[root(node)].append(node)
    return tuple(tuple(nodes) for nodes in groups.values())


def _table(contacts, beacon_counts, signatures):
    types = sorted(set(signatures))
    type_ids = [types.index(sig) for sig in signatures]
    rows = []
    for mask in range(1 << len(contacts)):
        groups = _partition(contacts, len(beacon_counts), mask)
        sums = tuple(sum(beacon_counts[i] for i in group) for group in groups)
        quota = tuple(sum(bool(mask & (1 << j)) for j, t in enumerate(type_ids) if t == i)
                      for i in range(len(types)))
        rows.append((mask, groups, quota, all(n == 1 for n in sums)))
    return types, type_ids, rows


def _paint(board, slots, mask, wall, floor):
    result = board.copy()
    for i, cells in enumerate(slots):
        color = floor if mask & (1 << i) else wall
        for cell in cells:
            result[cell] = color
    return result


def _candidates(inp):
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 4, "Expected background, wall, floor, and beacon")
    background = int(colors[np.argmax(counts)])
    _require(int(counts.max()) * 2 > inp.size, "Background must occupy more than half the canvas")
    area = int(np.count_nonzero(inp != background))
    ch, cw = inp.shape
    # Conservation fixes board area, and its intact uniform rim fixes its crop.
    for h in range(5, ch + 1):
        if area % h:
            continue
        w = area // h
        if not 5 <= w <= cw:
            continue
        for r in range(ch - h + 1):
            for c in range(cw - w + 1):
                wall = int(inp[r, c])
                if wall == background:
                    continue
                board = inp[r:r + h, c:c + w]
                if not (np.all(board[0] == wall) and np.all(board[-1] == wall)
                        and np.all(board[:, 0] == wall) and np.all(board[:, -1] == wall)):
                    continue
                outside = inp != background
                outside[r:r + h, c:c + w] = False
                exterior = _components(outside)
                if not 2 <= len(exterior) <= 10:
                    continue
                if any(len({int(inp[cell]) for cell in cells}) != 1 for cells in exterior):
                    continue
                exterior_colors = {int(inp[cells[0]]) for cells in exterior}
                if len(exterior_colors) != 2 or wall not in exterior_colors:
                    continue
                floor = next(iter(exterior_colors - {wall}))
                marker_colors = set(map(int, colors)) - {background, wall, floor}
                if len(marker_colors) != 1:
                    continue
                beacon = next(iter(marker_colors))
                if not np.any(board == floor) or not 2 <= np.count_nonzero(board == beacon) <= 5:
                    continue
                slots = _components(board == background)
                if len(slots) != len(exterior) or any(len(s) > 7 for s in slots):
                    continue
                signatures = [_normalize(s) for s in slots]
                inventory = Counter((int(inp[s[0]]), _normalize(s)) for s in exterior)
                if Counter(signatures) != Counter(_normalize(s) for s in exterior):
                    continue
                walls = (board == wall) | (board == background)
                beacon_cells = set(map(tuple, np.argwhere(board == beacon)))
                try:
                    regions, contacts, beacon_counts = _model(walls, slots, beacon_cells)
                except ValueError:
                    continue
                if any(n > 1 for n in beacon_counts):
                    continue
                types, type_ids, rows = _table(contacts, beacon_counts, signatures)
                quota = tuple(inventory[floor, sig] for sig in types)
                solutions = [row[0] for row in rows if row[2] == quota and row[3]]
                if not solutions:
                    continue
                mask = solutions[0]
                yield {"background": background, "wall": wall, "floor": floor, "beacon": beacon,
                       "box": (r, c, h, w), "board": board, "slots": slots,
                       "signatures": signatures, "inventory": inventory, "exterior": exterior,
                       "regions": regions, "contacts": contacts, "beacon_counts": beacon_counts,
                       "types": types, "type_ids": type_ids, "rows": rows, "quota": quota,
                       "mask": mask, "solution_count": len(solutions),
                       "output": _paint(board, slots, mask, wall, floor)}


def _decode(input_grid):
    inp = _grid(input_grid)
    candidates = list(_candidates(inp))
    _require(len(candidates) == 1, "Input must have one uniquely repairable board")
    _require(candidates[0]["solution_count"] == 1, "Inventory and beacons admit multiple repairs")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the crop, inventory, plate orientations, and seeded rooms."""
    return _decode(input_grid)["output"].copy()


def _witnesses(info):
    rows, mask, quota = info["rows"], info["mask"], info["quota"]
    valid = [row for row in rows if row[3]]
    exact = [row for row in rows if row[2] == quota]
    count = sum(info["beacon_counts"])
    area = sum(len(sig) * n for sig, n in zip(info["types"], quota))
    return {
        "without_rooms": len(exact) > 1,
        "without_inventory": len(valid) > 1,
        "count_only": any(row[0] != mask and len(row[1]) == count for row in exact),
        "area_only": any(row[0] != mask and sum(len(sig) * n for sig, n in
                         zip(info["types"], row[2])) == area for row in valid),
    }


def _wall_layout(rng):
    h, w = map(int, rng.integers(10, 19, size=2))
    boxes = [(0, 0, h - 1, w - 1)]
    target = int(rng.integers(6, 12))
    for _ in range(target - 1):
        choices = []
        for index, (t, l, b, r) in enumerate(boxes):
            for cut in range(l + 3, r - 2):
                choices.append((index, ((t, l, b, cut), (t, cut, b, r))))
            for cut in range(t + 3, b - 2):
                choices.append((index, ((t, l, cut, r), (cut, l, b, r))))
        if not choices:
            break
        index, halves = choices[int(rng.integers(len(choices)))]
        boxes[index:index + 1] = halves
    walls = np.zeros((h, w), dtype=bool)
    for t, l, b, r in boxes:
        walls[t, l:r + 1] = walls[b, l:r + 1] = True
        walls[t:b + 1, l] = walls[t:b + 1, r] = True
    return walls


def _plate_candidates(rng, walls):
    templates = {tuple((0, i) for i in range(n)) for n in range(1, 5)}
    templates.update(tuple((i, 0) for i in range(n)) for n in range(1, 5))
    # Procedural elbow, T and cross plates: independent arm lengths and arms.
    for _ in range(20):
        arms = rng.choice(4, size=int(rng.integers(2, 5)), replace=False)
        shape = {(0, 0)}
        for arm in arms:
            dr, dc = _DIRECTIONS[int(arm)]
            length = int(rng.integers(1, 3))
            shape.update((dr * k, dc * k) for k in range(1, length + 1))
        if len(shape) <= 7:
            templates.add(_normalize(shape))
    regions = _components(~walls)
    labels = np.full(walls.shape, -1, dtype=int)
    for index, cells in enumerate(regions):
        for cell in cells:
            labels[cell] = index
    candidates = []
    h, w = walls.shape
    for shape in sorted(templates):
        sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
        for r in range(1, h - sh):
            for c in range(1, w - sw):
                cells = frozenset((r + dr, c + dc) for dr, dc in shape)
                if not all(walls[cell] for cell in cells):
                    continue
                contacts = {int(labels[y + dy, x + dx]) for y, x in cells
                            for dy, dx in _DIRECTIONS if labels[y + dy, x + dx] >= 0}
                if len(contacts) >= 2:
                    candidates.append((shape, cells))
    rng.shuffle(candidates)
    return candidates


def _choose_repair(rng, walls, count, witness):
    candidates = _plate_candidates(rng, walls)
    slots, used, signatures = [], set(), []
    maximum_types = int(rng.choice([2, 3, 3, 4]))
    for shape, cells in candidates:
        if len(set(signatures) | {shape}) > maximum_types:
            continue
        if any((r + dr, c + dc) in used for r, c in cells
               for dr, dc in _DIRECTIONS + ((0, 0),)):
            continue
        slots.append(tuple(sorted(cells)))
        signatures.append(shape)
        used.update(cells)
        if len(slots) == count:
            break
    if len(slots) != count:
        return None
    regions, contacts, _ = _model(walls, slots, set())
    # The room partitions are independent of beacon placement: enumerate once.
    types, type_ids, rows = _table(contacts, [0] * len(regions), signatures)
    multiplicities = Counter(type_ids)
    for _ in range(36):
        m = int(rng.integers(2, min(5, len(regions)) + 1))
        marked = set(map(int, rng.choice(len(regions), size=m, replace=False)))
        valid = [row for row in rows if all(sum(i in marked for i in group) == 1 for group in row[1])]
        if len(valid) < 2:
            continue
        buckets = Counter(row[2] for row in valid)
        order = rng.permutation(len(valid))
        for index in order:
            mask, _, quota, _ = valid[int(index)]
            if buckets[quota] != 1 or mask in (0, (1 << count) - 1):
                continue
            if prod(comb(multiplicities[i], quota[i]) for i in range(len(types))) < 2:
                continue
            if witness == "count_only" and not any(row[2] == quota and row[0] != mask
                                                   and len(row[1]) == m for row in rows):
                continue
            area = sum(len(sig) * n for sig, n in zip(types, quota))
            if witness == "area_only" and not any(row[0] != mask and sum(len(sig) * n for sig, n
                    in zip(types, row[2])) == area for row in valid):
                continue
            beacons = []
            for i in sorted(marked):
                cells = regions[i]
                beacons.append(cells[int(rng.integers(len(cells)))])
            return slots, signatures, mask, beacons
    return None


def _scatter(rng, board, pieces, background):
    h, w = board.shape
    for attempt in range(16):
        ch = int(rng.integers(max(18, h + 4), 31)) if attempt < 12 else 30
        cw = int(rng.integers(max(18, w + 4), 31)) if attempt < 12 else 30
        if ch * cw <= 2 * h * w:
            continue
        r, c = int(rng.integers(ch - h + 1)), int(rng.integers(cw - w + 1))
        canvas = np.full((ch, cw), background, dtype=np.int8)
        canvas[r:r + h, c:c + w] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[max(0, r - 1):min(ch, r + h + 1), max(0, c - 1):min(cw, c + w + 1)] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i][1]), float(rng.random())))
        for index in order:
            color, shape = pieces[index]
            sh, sw = max(y for y, _ in shape) + 1, max(x for _, x in shape) + 1
            sites = [(y, x) for y in range(ch - sh + 1) for x in range(cw - sw + 1)
                     if all(not blocked[y + dy, x + dx] for dy, dx in shape)]
            if not sites:
                break
            y, x = sites[int(rng.integers(len(sites)))]
            for dy, dx in shape:
                yy, xx = y + dy, x + dx
                canvas[yy, xx] = color
                for ar, ac in _DIRECTIONS + ((0, 0),):
                    if 0 <= yy + ar < ch and 0 <= xx + ac < cw:
                        blocked[yy + ar, xx + ac] = True
        else:
            return canvas
    return None


def _make_pair(rng, palette, slot_count, witness=None):
    background, wall, floor, beacon = map(int, palette[:4])
    for _ in range(240):
        walls = _wall_layout(rng)
        repair = _choose_repair(rng, walls, slot_count, witness)
        if repair is None:
            continue
        slots, signatures, mask, beacons = repair
        board = np.where(walls, wall, floor).astype(np.int8)
        for cell in beacons:
            board[cell] = beacon
        output = _paint(board, slots, mask, wall, floor)
        pieces = []
        for i, cells in enumerate(slots):
            pieces.append((floor if mask & (1 << i) else wall, signatures[i]))
            for cell in cells:
                board[cell] = background
        canvas = _scatter(rng, board, pieces, background)
        if canvas is None:
            continue
        # Whole-scene transforms preserve the supplied orientations of pieces.
        rotation = int(rng.integers(4))
        canvas, output = np.rot90(canvas, rotation).copy(), np.rot90(output, rotation).copy()
        if rng.random() < 0.5:
            canvas, output = np.fliplr(canvas).copy(), np.fliplr(output).copy()
        info = _decode(canvas)
        _require(np.array_equal(info["output"], output), "Construction and input-only reconstruction differ")
        return {"input": canvas, "output": output}
    raise RuntimeError("No qualifying repair found within 240 bounded proposals")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= int(value) <= high, f"Illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    local_palettes = bool(rng.random() < 0.7)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            witness = ("count_only" if index == 0 else "area_only" if index == 1 else None) if split == "train" else None
            slot_count = int(rng.integers(5, 8) if split == "train" else rng.integers(8, 11))
            pair_palette = rng.permutation(10) if local_palettes else palette
            for _ in range(8):
                pair = _make_pair(rng, pair_palette, slot_count, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Duplicate inputs after bounded retries")
    rng.shuffle(puzzle["train"])
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training_evidence, split_slots = set(), Counter(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(info["output"], out), "Output is not the unique repair")
            fixed = info["board"] != info["background"]
            _require(np.array_equal(out[fixed], info["board"][fixed]), "Fixed cells changed")
            _require(not np.any(out == info["background"]), "Unfilled plate")
            # Recheck actual output pixels with flood fill, not contracted graph.
            rooms = _components(out != info["wall"])
            _require(all(sum(out[cell] == info["beacon"] for cell in room) == 1 for room in rooms),
                     "Every actual enclosed room must contain one beacon")
            supplied = Counter(int(inp[cell]) for cells in info["exterior"] for cell in cells)
            used = Counter(int(out[cell]) for cells in info["slots"] for cell in cells)
            _require(supplied == used, "Inventory color area changed")
            placed = Counter((int(out[cells[0]]), _normalize(cells)) for cells in info["slots"])
            _require(placed == info["inventory"], "Oriented piece multiplicities changed")
            evidence = _witnesses(info)
            _require(evidence["without_rooms"] and evidence["without_inventory"], "Either inherited concept is redundant")
            if split == "train":
                training_evidence.update(k for k, v in evidence.items() if v)
            split_slots[split].append(len(info["slots"]))
    _require(training_evidence["count_only"] and training_evidence["area_only"], "Missing shortcut counterexamples")
    _require(max(split_slots["train"]) < min(split_slots["test"]), "Missing unresolved-plate count shift")


def self_test():
    """Bounded, in-memory coverage across exactly 42 primary seeds."""
    shapes, boards, inventories, topologies, palettes, plates, signatures = (set() for _ in range(7))
    witness_counts, counts = Counter(), Counter()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed == 0:
            first = puzzle
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                shapes.add(pair["input"].shape)
                boards.add(pair["output"].shape)
                inventories.add(tuple(sorted(info["inventory"].items())))
                topologies.add((len(info["regions"]), tuple(sorted(len(x) for x in info["contacts"])),
                                tuple(sorted(len(row) for row in _partition(info["contacts"], len(info["regions"]), info["mask"])))))
                palettes.add(tuple(info[k] for k in ("background", "wall", "floor", "beacon")))
                plates.update(info["signatures"])
                counts[len(info["slots"]) ] += 1
                witness_counts.update(k for k, v in _witnesses(info).items() if v)
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "Nested-list solve failed")
    _require(len(signatures) == 42 and len(shapes) >= 30 and len(boards) >= 25, "Insufficient structural variation")
    _require(len(inventories) >= 60 and len(topologies) >= 25 and len(plates) >= 12, "Insufficient inventory/topology variation")
    _require(set(counts) == set(range(5, 11)) and len(palettes) >= 30, "Counts or palette roles did not vary")
    _require(any(max(r for r, _ in s) and max(c for _, c in s) for s in plates), "No bent or branched plates")
    again = generate_puzzle(0)
    _require(all(_key(p[k]) == _key(q[k]) for s in first for p, q in zip(first[s], again[s])
                 for k in ("input", "output")), "Nondeterministic generation")
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100 + nt + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Dihedral and palette equivariance, including rotations of exterior pieces.
    permutation = np.array([8, 6, 1, 9, 4, 0, 7, 2, 5, 3], dtype=np.int8)
    for pair in first["train"] + first["test"]:
        for transform in (np.rot90, np.fliplr, np.transpose):
            _require(np.array_equal(solve(transform(pair["input"])), transform(pair["output"])), "Geometric equivariance failed")
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "Color equivariance failed")
    # Validate rejection of wrong labels and missing inventory; no files saved.
    for kind in ("label", "inventory"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if kind == "label":
            pair["output"][0, 0] = info["floor"]
        else:
            pair["input"][info["exterior"][0][0]] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupted puzzle accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes), "board_shapes": len(boards),
            "inventories": len(inventories), "topology_summaries": len(topologies),
            "oriented_plate_shapes": len(plates), "palettes": len(palettes),
            "slot_counts": dict(sorted(counts.items())), "witnesses": dict(witness_counts)}
