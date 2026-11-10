"""Stationary trail-to-tip locators address an assembly of nine whole tiles.

The public solver reads only its input. Construction renders outputs directly
from sampled tile assignments; validation resegments the returned input. No
parent code, data files, persistent caches, or execution traces are used.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
SLOTS = tuple(itertools.product(range(3), repeat=2))
# These are payload shapes, never destination addresses in this composition.
B_MASKS = (((0, 0), (0, 1), (1, 0)),
           ((0, 0), (0, 1), (0, 2), (1, 1)),
           ((0, 1), (0, 2), (1, 2)),
           ((0, 0), (1, 0), (2, 0), (1, 1)),
           ((0, 2), (1, 2), (2, 2), (1, 1)),
           ((1, 0), (2, 0), (2, 1)),
           ((1, 1), (2, 0), (2, 1), (2, 2)),
           ((1, 2), (2, 1), (2, 2)))
MASK_TO_DIRECTION = {frozenset(mask): direction
                     for mask, direction in zip(B_MASKS, DIRECTIONS)}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
             "grids must be 2D with sides in 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= array) & (array <= 9)), "colors outside 0..9")
    return array.astype(np.int8, copy=False)


def _fingerprint(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _components(mask):
    """Four-connected components, including those meeting the canvas edge."""
    height, width = mask.shape
    seen = ~mask.copy()
    components = []
    for rr, cc in np.argwhere(mask):
        start = int(rr), int(cc)
        if seen[start]:
            continue
        seen[start] = True
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(cells)
    return components


def _rooms_for_wall(grid, wall):
    rooms = []
    height, width = grid.shape
    for cells in _components(grid != wall):
        rows, cols = zip(*cells)
        r, c, last_r, last_c = min(rows), min(cols), max(rows), max(cols)
        if r == 0 or c == 0 or last_r == height - 1 or last_c == width - 1:
            continue
        h, w = last_r - r + 1, last_c - c + 1
        if h * w != len(cells) or (h, w) not in ((3, 7), (7, 3), (9, 9)):
            # Neighbouring frames can incidentally close an empty background
            # pocket. It is neither a source bay nor the assembly frame.
            continue
        # Rectangular free regions must have a complete one-cell frame.
        frame = grid[r - 1:r + h + 1, c - 1:c + w + 1]
        if not (np.all(frame[0] == wall) and np.all(frame[-1] == wall)
                and np.all(frame[:, 0] == wall) and np.all(frame[:, -1] == wall)):
            continue
        rooms.append((r, c, h, w))
    if len(rooms) < 10 or sum((h, w) == (9, 9) for r, c, h, w in rooms) != 1:
        return None
    target = next(room for room in rooms if room[2:] == (9, 9))
    background = int(grid[target[0], target[1]])
    rooms = [room for room in rooms if room == target or
             np.any(grid[room[0]:room[0] + room[2], room[1]:room[1] + room[3]] != background)]
    if len(rooms) != 10:
        return None
    return rooms


def _decode_locator(key, background):
    """Two collinear trail cells precede one terminal tip, at unit spacing."""
    points = np.argwhere(key != background)
    if len(points) == 0:
        return None
    _require(len(points) == 3, "locator must contain exactly three marked cells")
    colors, counts = np.unique(key[key != background], return_counts=True)
    _require(sorted(map(int, counts)) == [1, 2], "locator needs two trails and one tip")
    trail, tip = int(colors[np.argmax(counts)]), int(colors[np.argmin(counts)])
    trails = set(map(tuple, np.argwhere(key == trail)))
    tr, tc = map(int, np.argwhere(key == tip)[0])
    directions = [(dr, dc) for dr, dc in DIRECTIONS
                  if trails == {(tr - dr, tc - dc), (tr - 2 * dr, tc - 2 * dc)}]
    _require(len(directions) == 1, "locator has no unique trail-to-tip direction")
    return {"direction": directions[0], "trail": trail, "tip": tip}


def _read_input(input_grid):
    grid = _grid(input_grid)
    interpretations = []
    for color, count in zip(*np.unique(grid, return_counts=True)):
        # Even a heavily shared set of ten frames contains more than 100 cells.
        if count < 100:
            continue
        rooms = _rooms_for_wall(grid, int(color))
        if rooms is not None:
            interpretations.append((int(color), rooms))
    _require(len(interpretations) == 1, "input needs one uniquely framed bay system")
    wall, rooms = interpretations[0]
    target = next(room for room in rooms if room[2:] == (9, 9))
    r, c, h, w = target
    background = int(grid[r, c])
    _require(np.all(grid[r:r + h, c:c + w] == background) and background != wall,
             "assembly frame must have an empty uniform 9x9 interior")
    records, explained = [], np.zeros(grid.shape, bool)
    for r, c, h, w in rooms:
        explained[r - 1:r + h + 1, c - 1:c + w + 1] = True
        if (h, w) == (9, 9):
            continue
        ends = ((r, c), (r, c + 4)) if h == 3 else ((r, c), (r + 4, c))
        gap = grid[r:r + 3, c + 3] if h == 3 else grid[r + 3, c:c + 3]
        _require(np.all(gap == background), "bay separator is not background")
        patches = [grid[rr:rr + 3, cc:cc + 3] for rr, cc in ends]
        dense = [i for i, patch in enumerate(patches) if np.all(patch != background)]
        _require(len(dense) == 1, "each bay needs exactly one fully occupied 3x3 tile")
        tile_index = dense[0]
        locator = _decode_locator(patches[1 - tile_index], background)
        direction = (0, 0) if locator is None else locator["direction"]
        records.append({"source": ends[tile_index], "key": ends[1 - tile_index],
                        "room": (r, c, h, w), "tile": patches[tile_index].copy(),
                        "direction": direction, "locator": locator,
                        "slot": (direction[0] + 1, direction[1] + 1)})
    _require(np.all(grid[~explained] == background), "unexplained cells outside frames")
    _require(len(records) == 9 and {record["slot"] for record in records} == set(SLOTS),
             "need one tile for each of nine slots, including the unkeyed center")
    center = next(record for record in records if record["locator"] is None)
    carrier = int(center["tile"][0, 0])
    _require(np.all(center["tile"] == carrier), "unkeyed center tile must be plain carrier")
    locators = [record["locator"] for record in records if record["locator"] is not None]
    trails, tips = {key["trail"] for key in locators}, {key["tip"] for key in locators}
    _require(len(trails) == len(tips) == 1, "locator roles must agree within an input")
    trail, tip = next(iter(trails)), next(iter(tips))
    reserved = {background, wall, carrier, trail, tip}
    _require(len(reserved) == 5, "background, wall, carrier, trail and tip must differ")
    for record in records:
        tile = record["tile"]
        _require(not np.any(np.isin(tile, [background, wall, trail, tip])),
                 "payload uses a reserved non-payload role")
        if record["locator"] is not None:
            _require(1 <= np.count_nonzero(tile != carrier) <= 6,
                     "outer tiles need 1..6 accent cells and at least three carrier cells")
            _require(1 <= len(set(map(int, tile.flat)) - {carrier}) <= 2,
                     "outer tiles need one or two accent colors")
    _require(len({record["tile"].tobytes() for record in records}) == 9,
             "all nine payloads must be distinguishable")
    return {"grid": grid, "wall": wall, "background": background, "carrier": carrier,
            "trail": trail, "tip": tip, "target": target[:2], "records": records}


def _render(info, reverse=False, accents_only=False, leave_sources=False):
    output = info["grid"].copy()
    if not leave_sources:
        for record in info["records"]:
            r, c = record["source"]
            output[r:r + 3, c:c + 3] = info["background"]
    tr, tc = info["target"]
    for record in info["records"]:
        dr, dc = record["direction"]
        slot = (1 - dr, 1 - dc) if reverse else record["slot"]
        r, c = tr + 3 * slot[0], tc + 3 * slot[1]
        tile = record["tile"]
        if accents_only:
            output[r:r + 3, c:c + 3][tile != info["carrier"]] = tile[tile != info["carrier"]]
        else:
            output[r:r + 3, c:c + 3] = tile
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate all nine full tiles to their locally decoded assembly slots."""
    return _render(_read_input(input_grid))


def _window_sums(mask, h, w):
    integral = np.pad(mask.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def _legal_positions(shape, rectangles, h, w):
    occupied, interiors = np.zeros(shape, bool), np.zeros(shape, bool)
    for r, c, old_h, old_w in rectangles:
        occupied[r:r + old_h, c:c + old_w] = True
        interiors[r + 1:r + old_h - 1, c + 1:c + old_w - 1] = True
    # Only boundary/boundary overlap is legal. Closed rooms may share walls.
    whole_free = _window_sums(interiors, h, w) == 0
    inside_free = _window_sums(occupied, h - 2, w - 2)[1:-1, 1:-1] == 0
    return np.argwhere(whole_free & inside_free)


def _layout(rng, relocate=True):
    """Start with guaranteed capacity, then relocate and turn rooms in place.

    All searches are over at most 900 cells. The current placement is always a
    valid fallback, so packing has no unbounded rejection tail.
    """
    height, width = map(int, rng.integers(25, 31, size=2))
    off_r, off_c = int(rng.integers(height - 24)), int(rng.integers(width - 24))
    top_vertical = [(off_r, off_c + col, 9, 5) for col in (12, 16, 20)]
    bottom = [(off_r + row, off_c + col, 5, 9)
              for row in (12, 16, 20) for col in (0, 8, 16)]
    vertical_count = int(rng.integers(1, 4))
    bays = [top_vertical[int(i)] for i in rng.permutation(3)[:vertical_count]]
    bays += [bottom[int(i)] for i in rng.permutation(9)[:9 - vertical_count]]
    rectangles = bays + [(off_r, off_c, 11, 11)]
    for _ in range(3 if relocate else 0):
        for index in map(int, rng.permutation(10)):
            old = rectangles[index]
            h, w = old[2:]
            if index < 9 and rng.random() < .45:
                h, w = w, h
            candidates = _legal_positions((height, width),
                                          rectangles[:index] + rectangles[index + 1:], h, w)
            if len(candidates):
                r, c = map(int, candidates[int(rng.integers(len(candidates)))])
                rectangles[index] = r, c, h, w
    return (height, width), rectangles[:9], rectangles[9]


def _locator(rng, direction, background, trail, tip):
    dr, dc = direction
    ends = [(r, c) for r, c in itertools.product(range(3), repeat=2)
            if 0 <= r - 2 * dr < 3 and 0 <= c - 2 * dc < 3]
    r, c = ends[int(rng.integers(len(ends)))]
    key = np.full((3, 3), background, np.int8)
    key[r, c] = tip
    key[r - dr, c - dc] = key[r - 2 * dr, c - 2 * dc] = trail
    return key


def _payloads(rng, carrier, accents, simple, multi_count):
    result = {(0, 0): np.full((3, 3), carrier, np.int8)}
    used = {result[(0, 0)].tobytes()}
    # A derangement makes every B-shaped training payload contradict the old
    # B address convention. In the new rule the stationary locator has authority.
    permutation = np.roll(np.arange(8), int(rng.integers(1, 8)))
    multi_indices = set(map(int, rng.permutation(8)[:multi_count]))
    for index, direction in enumerate(DIRECTIONS):
        for attempt in range(100):
            tile = np.full((3, 3), carrier, np.int8)
            if simple:
                mask = list(B_MASKS[int(permutation[index])])
            elif index <= 1:
                # Two guaranteed asymmetric motifs with conflicting addresses.
                mask = list(B_MASKS[int(permutation[index])])
            else:
                size = int(rng.integers(2 if index in multi_indices else 1, 7))
                mask = [divmod(int(i), 3) for i in rng.choice(9, size, replace=False)]
            color = int(rng.choice(accents))
            for point in mask:
                tile[point] = color
            if index in multi_indices:
                second = int(rng.choice([x for x in accents if x != color]))
                selected = rng.choice(len(mask), int(rng.integers(1, len(mask))), replace=False)
                for i in selected:
                    tile[mask[int(i)]] = second
            if tile.tobytes() not in used:
                result[direction] = tile
                used.add(tile.tobytes())
                break
        else:
            raise RuntimeError("bounded distinct-payload construction exhausted")
    return result


def _draw_frame(grid, rectangle, wall):
    r, c, h, w = rectangle
    grid[r, c:c + w] = grid[r + h - 1, c:c + w] = wall
    grid[r:r + h, c] = grid[r:r + h, c + w - 1] = wall


def _make_pair(rng, palette, simple, multi_count, relocate=True):
    background, wall, carrier, trail, tip = map(int, palette[:5])
    accents = list(map(int, palette[5:]))
    shape, bays, target = _layout(rng, relocate=relocate)
    tiles = _payloads(rng, carrier, accents, simple, multi_count)
    a = np.full(shape, background, np.int8)
    for rectangle in bays + [target]:
        _draw_frame(a, rectangle, wall)
    assigned = list(tiles)
    assigned = [assigned[int(i)] for i in rng.permutation(9)]
    sources = []
    for (r, c, h, w), direction in zip(bays, assigned):
        ends = ((r + 1, c + 1), (r + 1, c + 5)) if h == 5 else ((r + 1, c + 1), (r + 5, c + 1))
        tile_index = int(rng.integers(2))
        sr, sc = ends[tile_index]
        kr, kc = ends[1 - tile_index]
        a[sr:sr + 3, sc:sc + 3] = tiles[direction]
        sources.append((sr, sc))
        if direction != (0, 0):
            a[kr:kr + 3, kc:kc + 3] = _locator(rng, direction, background, trail, tip)
    b = a.copy()
    for r, c in sources:
        b[r:r + 3, c:c + 3] = background
    tr, tc = target[0] + 1, target[1] + 1
    # Construct labels from sampled assignments, not by calling solve().
    for (dr, dc), tile in tiles.items():
        r, c = tr + 3 * (dr + 1), tc + 3 * (dc + 1)
        b[r:r + 3, c:c + 3] = tile
    turns = int(rng.integers(4))
    a, b = np.rot90(a, turns), np.rot90(b, turns)
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    return {"input": a.copy(), "output": b.copy()}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    stable_palette = bool(rng.integers(3) == 0)
    train_modes = [(True, 0), (False, int(rng.integers(2, 5)))]
    train_modes += [(False, int(rng.integers(0, 5))) for _ in range(num_train - 2)]
    rng.shuffle(train_modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            simple, multi = train_modes[index] if split == "train" else (False, int(rng.integers(5, 9)))
            local_palette = palette if stable_palette else rng.permutation(10)
            for attempt in range(8):
                pair = _make_pair(rng, local_palette, simple, multi, relocate=attempt < 7)
                try:
                    # Exclude accidentally enclosed empty 9x9 pockets that
                    # could visually compete with the intended target frame.
                    _read_input(pair["input"])
                except AssertionError:
                    continue
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    return puzzle


def _measure(info):
    multi = sum(len(set(map(int, record["tile"].flat)) - {info["carrier"]}) > 1
                for record in info["records"])
    conflicts, asymmetries, masks = 0, 0, set()
    for record in info["records"]:
        tile = record["tile"]
        mask = frozenset(map(tuple, np.argwhere(tile != info["carrier"])))
        masks.add(tuple(sorted(mask)))
        old_direction = MASK_TO_DIRECTION.get(mask)
        conflicts += old_direction is not None and old_direction != record["direction"]
        asymmetries += not np.array_equal(tile, np.rot90(tile))
    return {"multi": multi, "conflicts": conflicts, "asymmetries": asymmetries,
            "masks": masks, "shape": info["grid"].shape,
            "roles": tuple(info[k] for k in ("background", "wall", "carrier", "trail", "tip")),
            "horizontal": sum(record["room"][2] == 3 for record in info["records"])}


def validate_generated_puzzle(puzzle):
    """Reconstruct and check geometry, conservation, parent necessity and evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, measurements = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "canvas must be preserved")
            key = _fingerprint(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _read_input(a)
            _require(np.array_equal(_render(info), b), "output violates locator-addressed tile translation")
            _require(np.array_equal(np.bincount(a.ravel(), minlength=10),
                                    np.bincount(b.ravel(), minlength=10)), "color mass is not conserved")
            mutable = np.zeros(a.shape, bool)
            tr, tc = info["target"]
            mutable[tr:tr + 9, tc:tc + 9] = True
            for record in info["records"]:
                r, c = record["source"]
                _require(np.all(b[r:r + 3, c:c + 3] == info["background"]), "source tile not cleared")
                mutable[r:r + 3, c:c + 3] = True
                kr, kc = record["key"]
                _require(np.array_equal(a[kr:kr + 3, kc:kc + 3], b[kr:kr + 3, kc:kc + 3]),
                         "stationary locator changed")
            _require(np.count_nonzero(a != b) == 162, "exactly 81 source and 81 target cells must change")
            _require(np.array_equal(a[~mutable], b[~mutable]), "walls or unrelated cells changed")
            _require(not np.any(b[tr:tr + 9, tc:tc + 9] == info["background"]), "assembly has holes")
            _require(not np.array_equal(_render(info, reverse=True), b), "locator polarity is unnecessary")
            _require(not np.array_equal(_render(info, accents_only=True), b), "full-tile copying is unnecessary")
            _require(not np.array_equal(_render(info, leave_sources=True), b), "clearing sources is unnecessary")
            measured = _measure(info)
            _require(measured["conflicts"] >= 2, "need payload motifs contradicting B-only addressing")
            _require(measured["asymmetries"] >= 2, "need evidence distinguishing translation from rotation")
            measurements[split].append(measured)
    train = measurements["train"]
    _require(any(m["multi"] == 0 and m["conflicts"] >= 6 for m in train),
             "need an easy training witness against accent-shape addressing")
    _require(any(2 <= m["multi"] <= 4 for m in train), "need a multi-accent whole-tile demonstration")
    _require(all(m["multi"] <= 4 for m in train), "training payload-complexity range violated")
    _require(all(5 <= m["multi"] <= 8 for m in measurements["test"]), "test complexity shift missing")


def self_test():
    """42 streamed seeds, every legal count, metamorphic and corruption checks.

    No files are written. One puzzle is retained at a time; only hashes and
    bounded sets of shapes/masks/role tuples survive each iteration.
    """
    hashes, shapes, masks, roles, layouts, multi_counts, orientations = set(), set(), set(), set(), set(), set(), set()
    normalized, pairs = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                info = _read_input(a)
                measured = _measure(info)
                pairs += 1
                shapes.add(a.shape)
                masks.update(measured["masks"])
                roles.add(measured["roles"])
                multi_counts.add(measured["multi"])
                orientations.add(measured["horizontal"])
                layouts.add((a.shape, info["target"],
                             tuple((r["source"], r["key"], r["direction"]) for r in info["records"])))
                mapping = {}
                normalized.add(hashlib.sha256(bytes(mapping.setdefault(int(x), len(mapping)) for x in a.flat)).hexdigest())
                _require(np.array_equal(solve(a), b), "input-only solve failed")
        hashes.add(digest.digest())
    _require(len(hashes) == 42 and pairs == 210, "42-seed puzzle diversity failed")
    _require(len(shapes) >= 20 and len(layouts) >= 180, "insufficient geometric diversity")
    _require(len(masks) >= 100 and len(roles) >= 50, "insufficient payload/palette diversity")
    _require(len(normalized) >= 200 and len(orientations) >= 6, "insufficient structural variation")
    _require({0, 2, 3, 4, 5, 6, 7, 8} <= multi_counts, "missing payload complexity regimes")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, nq), "explicit counts ignored")
            validate_generated_puzzle(puzzle)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(1729, 2, 1), generate_puzzle(1729, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in first
                 for p, q in zip(first[s], second[s]) for k in ("input", "output")), "nondeterminism")
    pair = first["test"][0]
    a, b = pair["input"], pair["output"]
    for turns in (1, 2, 3):
        _require(np.array_equal(solve(np.rot90(a, turns)), np.rot90(b, turns)), "rotation equivariance")
    _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(b)), "reflection equivariance")
    recolor = np.random.default_rng(99).permutation(10)
    _require(np.array_equal(solve(recolor[a]), recolor[b]), "color-role equivariance")
    info = _read_input(a)
    swapped = a.copy()
    outer = [r for r in info["records"] if r["locator"] is not None]
    for dst, src in ((outer[0], outer[1]), (outer[1], outer[0])):
        r, c = dst["key"]
        rr, cc = src["key"]
        swapped[r:r + 3, c:c + 3] = a[rr:rr + 3, cc:cc + 3]
    moved = solve(swapped)
    _require(not np.array_equal(moved, b), "changing locator assignment must change assembly")
    # Entire payloads at the two destinations must exchange, with keys fixed.
    tr, tc = info["target"]
    for dst, src in ((outer[0], outer[1]), (outer[1], outer[0])):
        r, c = tr + 3 * dst["slot"][0], tc + 3 * dst["slot"][1]
        _require(np.array_equal(moved[r:r + 3, c:c + 3], src["tile"]), "locator swap did not exchange full tiles")
    corruptions = [a.copy(), b.copy(), b.copy()]
    # A missing carrier cell and a rotated asymmetric tile can preserve much
    # of an otherwise correct answer; validation must still reject both.
    corruptions[1][tr + 3, tc + 3] = info["background"]
    witness = next(r for r in outer if not np.array_equal(r["tile"], np.rot90(r["tile"])))
    wr, wc = tr + 3 * witness["slot"][0], tc + 3 * witness["slot"][1]
    corruptions[2][wr:wr + 3, wc:wc + 3] = np.rot90(witness["tile"])
    for corrupted in corruptions:
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        bad["test"][0]["output"] = corrupted
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted an output corruption")
