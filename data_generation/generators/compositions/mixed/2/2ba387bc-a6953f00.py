"""Parity-selected donor crops inserted into independently ranked frames.

Only NumPy and the standard library are used. No files, examples, seeds or
generation metadata are consulted by solve(). All semantic conventions are
fixed; random choices control only the distribution of visible inputs.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


TILE = 4  # A one-cell rim surrounding an exact 2x2 receiving cavity.


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, minimum, maximum=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
            or (maximum is not None and int(value) > maximum)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= d <= 30 for d in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return array


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _parse_background(grid, background):
    """Binary connected components treat multicolor donors as single objects."""
    todo = grid != background
    objects = []
    height, width = grid.shape
    for r, c in np.argwhere(todo):
        r, c = int(r), int(c)
        if not todo[r, c]:
            continue
        todo[r, c] = False
        stack = [(r, c)]
        top = bottom = r
        left = right = c
        while stack:
            rr, cc = stack.pop()
            top, bottom = min(top, rr), max(bottom, rr)
            left, right = min(left, cc), max(right, cc)
            if bottom - top >= TILE or right - left >= TILE:
                raise ValueError("component exceeds tile footprint")
            for nr, nc in ((rr - 1, cc), (rr + 1, cc),
                           (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and todo[nr, nc]:
                    todo[nr, nc] = False
                    stack.append((nr, nc))
        if bottom - top != TILE - 1 or right - left != TILE - 1:
            raise ValueError("component is not a 4x4 tile")
        tile = grid[top:bottom + 1, left:right + 1].astype(np.int8, copy=True)
        if np.all(tile != background):
            kind = "S"
        else:
            rim = np.concatenate((tile[0], tile[-1], tile[1:-1, 0], tile[1:-1, -1]))
            if (rim[0] == background or not np.all(rim == rim[0])
                    or not np.all(tile[1:-1, 1:-1] == background)):
                raise ValueError("invalid receiving frame")
            kind = "H"
        neighborhood = grid[max(0, top - 1):min(height, bottom + 2),
                            max(0, left - 1):min(width, right + 2)].copy()
        neighborhood[top - max(0, top - 1):bottom + 1 - max(0, top - 1),
                     left - max(0, left - 1):right + 1 - max(0, left - 1)] = background
        if not np.all(neighborhood == background):
            raise ValueError("tiles must have a one-cell background gap")
        objects.append(dict(top=top, left=left, kind=kind, tile=tile))
    queues = [sorted((o for o in objects if o["kind"] == kind),
                     key=lambda o: (o["top"], o["left"])) for kind in ("H", "S")]
    if not 1 <= len(queues[0]) == len(queues[1]) <= 7:
        raise ValueError("need equal positive counts, at most seven per class")
    return int(background), queues[0], queues[1]


def _analyze(input_grid):
    grid = _grid(input_grid)
    if grid.shape[0] != grid.shape[1]:
        raise ValueError("canvas must be square")
    interpretations = []
    for background in np.unique(grid):
        try:
            interpretations.append(_parse_background(grid, int(background)))
        except ValueError:
            pass
    if len(interpretations) != 1:
        raise ValueError("input has no unique geometric background interpretation")
    return interpretations[0]


def _render(background, hollows, solids, left):
    out = np.full((TILE * len(hollows), 2 * TILE), background, dtype=np.int8)
    for rank, (hollow, solid) in enumerate(zip(hollows, solids)):
        row = TILE * rank
        out[row:row + TILE, :TILE] = hollow["tile"]
        out[row:row + TILE, TILE:] = solid["tile"]
        crop = solid["tile"][:2, :2] if left else solid["tile"][:2, -2:]
        out[row + 1:row + 3, 1:3] = crop
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output from visible geometry and whole-canvas parity."""
    grid = _grid(input_grid)
    background, hollows, solids = _analyze(grid)
    return _render(background, hollows, solids, bool(grid.shape[0] % 2))


def _starts(rng, side, amount):
    extra = side - (5 * amount - 1)
    gaps = rng.multinomial(extra, np.full(amount + 1, 1 / (amount + 1)))
    starts = [int(gaps[0])]
    for i in range(amount - 1):
        starts.append(starts[-1] + 5 + int(gaps[i + 1]))
    return starts


def _positions(rng, count, side, witness=False):
    """A feasible square packing followed by free, separation-preserving moves."""
    capacity = (side + 1) // 5
    rows = _starts(rng, side, capacity)
    columns = [_starts(rng, side, capacity) for _ in rows]
    slots = [(r, c) for r in range(capacity) for c in range(capacity)]
    selected = []
    locked = set()
    if witness:
        # Independently demonstrate within-class ties for BOTH queues.
        bands = rng.choice(capacity, size=2, replace=False)
        for band in bands:
            cols = rng.choice(capacity, size=2, replace=False)
            selected.extend((int(band), int(c)) for c in cols)
        remaining = [p for p in slots if p not in selected]
        picks = rng.choice(len(remaining), size=2 * count - 4, replace=False)
        rest = [remaining[int(i)] for i in picks]
        # H0,H1 and S0,S1 start at tied rows. The other objects are unconstrained.
        selected = selected[:2] + rest[:count - 2] + selected[2:] + rest[count - 2:]
        locked = {0, 1, count, count + 1}
    else:
        selected = [slots[int(i)] for i in rng.choice(len(slots), size=2 * count, replace=False)]
    positions = [(rows[r], columns[r][c]) for r, c in selected]
    rr, cc = np.indices((side - 3, side - 3))
    for _ in range(3):
        for index in rng.permutation(2 * count):
            index = int(index)
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if j != index:
                    legal &= (np.abs(rr - r) > 4) | (np.abs(cc - c) > 4)
            if index in locked:
                legal &= rr == positions[index][0]
            sites = np.argwhere(legal)
            point = sites[int(rng.integers(len(sites)))]
            positions[index] = tuple(int(x) for x in point)
    if rng.random() < .45:
        edge = int(rng.integers(4))
        dr = (-min(r for r, c in positions) if edge == 0 else
              side - 4 - max(r for r, c in positions) if edge == 1 else 0)
        dc = (-min(c for r, c in positions) if edge == 2 else
              side - 4 - max(c for r, c in positions) if edge == 3 else 0)
        positions = [(r + dr, c + dc) for r, c in positions]
    return positions


def _donor(rng, palette):
    k = int(rng.integers(1, len(palette) + 1))
    colors = rng.choice(palette, size=k, replace=False)
    rows, cols = np.indices((4, 4))
    mode = int(rng.integers(7))
    if mode == 0:
        labels = rng.integers(k, size=(4, 4))
    elif mode == 1:
        labels = np.full((4, 4), int(rng.integers(k)))
    elif mode == 2:
        labels = np.broadcast_to(rng.integers(k, size=(1, 4)), (4, 4))
    elif mode == 3:
        base = rng.integers(k, size=(2, 2))
        labels = base[rows // 2, cols // 2]
    elif mode == 4:
        base = rng.integers(k, size=(2, 2))
        labels = base[rows % 2, cols % 2]
    elif mode == 5:
        labels = np.zeros((4, 4), dtype=int)
        labels.flat[rng.choice(16, size=int(rng.integers(1, 6)), replace=False)] = rng.integers(k)
    else:
        labels = (rows * int(rng.integers(1, k + 1)) + cols + int(rng.integers(k))) % k
    return colors[labels].astype(np.int8, copy=True)


def _draw(rng, side, count, background, positions, witness):
    palette = np.array([c for c in range(10) if c != background], dtype=np.int8)
    frame_colors = (rng.permutation(palette)[:count] if witness else
                    rng.choice(palette, size=count, replace=True))
    grid = np.full((side, side), background, dtype=np.int8)
    for index, (r, c) in enumerate(positions):
        if index < count:
            tile = np.full((4, 4), frame_colors[index], dtype=np.int8)
            tile[1:3, 1:3] = background
        else:
            rank = index - count
            tile = _donor(rng, palette)
            if witness:
                # Unique crops on each side and differing left/right crops.
                tile[0, 0] = frame_colors[rank]
                tile[0, 2] = frame_colors[(rank + 1) % count]
                # Literal receiver colors also occur in fully occupied donors.
                tile[3, 3] = frame_colors[rank]
        grid[r:r + 4, c:c + 4] = tile
    return grid


def _greedy_rows(hollows, solids):
    occupied, row = set(), 0
    for obj in sorted(hollows + solids, key=lambda o: (o["top"], o["left"])):
        if obj["kind"] in occupied:
            row += 1
            occupied = set()
        occupied.add(obj["kind"])
    return row + 1


def _nearest_output(analysis, left):
    background, hollows, solids = analysis
    out = _render(background, hollows, solids, left)
    for rank, hollow in enumerate(hollows):
        donor = min(solids, key=lambda s: (abs(s["top"] - hollow["top"])
                                          + abs(s["left"] - hollow["left"]),
                                          s["top"], s["left"]))
        out[4 * rank + 1:4 * rank + 3, 1:3] = donor["tile"][:2, :2] if left else donor["tile"][:2, -2:]
    return out


def _tie_witness(queue):
    return any(a["top"] == b["top"] and not np.array_equal(a["tile"], b["tile"])
               for a, b in zip(queue, queue[1:]))


def _witness_ok(analysis, side):
    background, hollows, solids = analysis
    left = bool(side % 2)
    correct = _render(background, hollows, solids, left)
    if (_greedy_rows(hollows, solids) <= len(hollows)
            or not all(_tie_witness(q) for q in (hollows, solids))
            or np.array_equal(correct, _nearest_output(analysis, left))):
        return False
    for reverse in (False, True):
        queues = [sorted(q, key=lambda o: (o["left"], o["top"]), reverse=reverse)
                  for q in (hollows, solids)]
        if np.array_equal(correct, _render(background, *queues, left)):
            return False
    return True


def _first_scene(rng):
    count = int(rng.integers(3, 5))
    side = int(rng.integers(19, 27))
    background = int(rng.integers(10))
    for _ in range(24):
        positions = _positions(rng, count, side, witness=True)
        grid = _draw(rng, side, count, background, positions, witness=True)
        if _witness_ok(_parse_background(grid, background), side):
            return grid, count, background
    # A bounded constructive fallback with the same count, not an extra rule.
    h = [(0, 10), (0, 15), (10, 0), (10, 10)][:count]
    s = [(5, 0), (5, 10), (15, 15), (15, 0)][:count]
    grid = _draw(rng, side, count, background, h + s, witness=True)
    _require(_witness_ok(_parse_background(grid, background), side), "fallback witness")
    return grid, count, background


def _sample_side(rng, count, parity=None, exclude=()):
    minimum = 5 * math.ceil(math.sqrt(2 * count)) - 1
    candidates = [s for s in range(minimum, 31)
                  if (parity is None or s % 2 == parity) and s not in exclude]
    return int(rng.choice(candidates))


def _new_pair(rng, count, side, used):
    for _ in range(16):
        background = int(rng.integers(10))
        positions = _positions(rng, count, side)
        grid = _draw(rng, side, count, background, positions, witness=False)
        if _key(grid) not in used:
            used.add(_key(grid))
            return {"input": grid, "output": solve(grid)}
    raise RuntimeError("bounded uniqueness attempts exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    small, first_count, background = _first_scene(rng)
    big = np.full((small.shape[0] + 1, small.shape[1] + 1), background, dtype=np.int8)
    big[:-1, :-1] = small
    first = [small, big]
    if rng.integers(2):
        first.reverse()
    train = [{"input": g, "output": solve(g)} for g in first]
    used = {_key(g) for g in first}
    grow = bool(rng.random() < .65)
    for index in range(2, num_train):
        count = int(rng.integers(2, 5) if grow else rng.integers(3, 6))
        if index < 4:
            reference = first[index - 2].shape[0]
            side = _sample_side(rng, count, parity=reference % 2, exclude=(reference,))
        else:
            side = _sample_side(rng, count)
        train.append(_new_pair(rng, count, side, used))
    test = []
    first_test_parity = int(rng.integers(2))
    for index in range(num_test):
        count = int(rng.integers(5, 8) if grow else rng.integers(1, 3))
        parity = (first_test_parity + index) % 2
        side = _sample_side(rng, count, parity=parity)
        test.append(_new_pair(rng, count, side, used))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantics, evidence against shortcuts, counts and a novel-count shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split type")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    parity_witness = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            analysis = _analyze(grid)
            background, hollows, solids = analysis
            count = len(hollows)
            left = bool(grid.shape[0] % 2)
            expected = _render(background, hollows, solids, left)
            _require(np.array_equal(out, expected), "incorrect output")
            _require(out.shape == (4 * count, 8), "output shape")
            for rank, (h, s) in enumerate(zip(hollows, solids)):
                slot = out[4 * rank:4 * rank + 4, :4]
                rim = np.ones((4, 4), dtype=bool)
                rim[1:3, 1:3] = False
                _require(np.array_equal(slot[rim], h["tile"][rim]), "frame altered")
                _require(np.array_equal(out[4 * rank:4 * rank + 4, 4:], s["tile"]), "donor altered")
                _require(np.all(slot[1:3, 1:3] != background), "unfilled cavity")
            analyses[split].append((grid, analysis))
            if split == "train" and not np.array_equal(out, _render(background, hollows, solids, not left)):
                parity_witness.add(grid.shape[0] % 2)
    _require(parity_witness == {0, 1}, "missing parity branch witness")
    training = analyses["train"]
    _require(any(_witness_ok(a, g.shape[0]) for g, a in training), "missing rank/tie/nearest witness")
    # Omitting rank attachment, reversing donor ranks, and reversing either
    # class's tie order each visibly contradict at least one demonstration.
    for kind_index in (1, 2):
        def reversed_ties(a):
            q = a[kind_index]
            return sorted(q, key=lambda o: (o["top"], -o["left"]))
        _require(any(not np.array_equal(
            _render(*a, bool(g.shape[0] % 2)),
            _render(a[0], reversed_ties(a) if kind_index == 1 else a[1],
                    reversed_ties(a) if kind_index == 2 else a[2], bool(g.shape[0] % 2)))
            for g, a in training), "tie direction is not demonstrated")
    _require(any(not np.array_equal(_render(*a, bool(g.shape[0] % 2)),
                                   _render(a[0], a[1], list(reversed(a[2])), bool(g.shape[0] % 2)))
                 for g, a in training), "donor rank is not observable")
    # Matched scenes isolate WHOLE canvas parity from every object property.
    matched = False
    for i, (g, a) in enumerate(training):
        for other, b in training[i + 1:]:
            small, big = sorted((g, other), key=lambda x: x.shape[0])
            if big.shape[0] != small.shape[0] + 1 or a[0] != b[0]:
                continue
            if (np.array_equal(small, big[:-1, :-1])
                    and np.all(big[-1, :] == a[0]) and np.all(big[:, -1] == a[0])):
                matched = True
    _require(matched, "missing controlled canvas-parity demonstration")
    train_counts = [len(a[1]) for g, a in training]
    for g, a in analyses["test"]:
        _require(len(a[1]) < min(train_counts) or len(a[1]) > max(train_counts),
                 "test should shift rank count beyond training range")
    if n >= 4:
        for parity in (0, 1):
            _require(len({g.shape[0] for g, a in training if g.shape[0] % 2 == parity}) >= 2,
                     "need multiple canvas sizes per parity")
    if m >= 2:
        _require({g.shape[0] % 2 for g, a in analyses["test"]} == {0, 1}, "test parity coverage")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; keep only bounded sets of small signatures in memory."""
    signatures, structures, layouts = set(), set(), set()
    sides, counts, backgrounds, parities = set(), set(), set(), set()
    donor_patterns, equal_corners, edge_pairs, changed_backgrounds = set(), 0, 0, 0
    shifts, pair_total = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        local_backgrounds, local_counts = set(), []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                original = inp.copy()
                _require(np.array_equal(solve(inp), out), "solve disagreement")
                _require(np.array_equal(inp, original), "solve mutated input")
                bg, h, s = _analyze(inp)
                side = inp.shape[0]
                count = len(h)
                sides.add(side)
                counts.add(count)
                backgrounds.add(bg)
                parities.add((split, side % 2))
                local_backgrounds.add(bg)
                if split == "train":
                    local_counts.append(count)
                else:
                    shifts.add("grow" if count > max(local_counts) else "shrink")
                layout = (side, tuple((o["top"], o["left"], o["kind"]) for o in h + s))
                layouts.add(layout)
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape) + normalized).hexdigest())
                edge_pairs += any(o["top"] == 0 or o["left"] == 0 or o["top"] + 4 == side
                                  or o["left"] + 4 == side for o in h + s)
                for donor in s:
                    tile = donor["tile"]
                    equal_corners += np.array_equal(tile[:2, :2], tile[:2, -2:])
                    labels = {}
                    donor_patterns.add(bytes(labels.setdefault(int(c), len(labels)) for c in tile.flat))
                pair_total += 1
        changed_backgrounds += len(local_backgrounds) > 1
    _require(len(signatures) == 42, "puzzle diversity")
    _require(len(structures) >= 180 and len(layouts) >= 180, "structural diversity")
    _require(len(sides) >= 17 and counts == set(range(1, 8)), "size/count diversity")
    _require(backgrounds == set(range(10)) and changed_backgrounds >= 30, "background diversity")
    _require(parities == {("train", 0), ("train", 1), ("test", 0), ("test", 1)}, "parity diversity")
    _require(shifts == {"grow", "shrink"}, "shift diversity")
    _require(len(donor_patterns) >= 60 and equal_corners >= 10 and edge_pairs >= 20,
             "donor or boundary variation")
    legal_calls = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(0, train, test)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts")
            legal_calls += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (2.0, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative controls target content copying and evidence, not just shapes.
    puzzle = generate_puzzle(0)
    damaged = {split: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[split]]
               for split in ("train", "test")}
    damaged["test"][0]["output"][1, 1] = (int(damaged["test"][0]["output"][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt crop accepted")
    for split in ("train", "test"):
        for pair in puzzle[split]:
            bg, h, s = _analyze(pair["input"])
            no_transfer = _render(bg, h, s, bool(pair["input"].shape[0] % 2))
            no_transfer[1::4, 1:3] = bg
            no_transfer[2::4, 1:3] = bg
            _require(not np.array_equal(no_transfer, pair["output"]), "receiver transfer is inert")
    return {"seeds": 42, "pairs": pair_total, "distinct_puzzles": len(signatures),
            "normalized_inputs": len(structures), "layouts": len(layouts),
            "canvas_sides": sorted(sides), "rank_counts": sorted(counts),
            "backgrounds": sorted(backgrounds), "donor_patterns": len(donor_patterns),
            "equal_corner_donors": int(equal_corners), "edge_pairs": int(edge_pairs),
            "count_shifts": sorted(shifts), "legal_count_calls": legal_calls,
            "semantic_verification": "generator-owned checks only"}
