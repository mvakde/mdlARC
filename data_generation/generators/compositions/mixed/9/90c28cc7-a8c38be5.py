"""Normalize expanded mosaic tiles, then assemble by their accent addresses.

Only the plain center has invisible logical seams: its 1x1 run-collapse is
restored to the family's fixed 3x3 carrier tile. See composition.md.
"""
from __future__ import annotations

import hashlib
import numpy as np


_SLOTS = tuple((r, c) for r in range(3) for c in range(3))
_CENTER = (1, 1)
_DIRECTIONS = tuple(s for s in _SLOTS if s != _CENTER)
_CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
_MASKS = {
    (0, 0): ((0, 0), (0, 1), (1, 0)),
    (0, 1): ((0, 0), (0, 1), (0, 2), (1, 1)),
    (0, 2): ((0, 1), (0, 2), (1, 2)),
    (1, 0): ((0, 0), (1, 0), (2, 0), (1, 1)),
    (1, 2): ((0, 2), (1, 1), (1, 2), (2, 2)),
    (2, 0): ((1, 0), (2, 0), (2, 1)),
    (2, 1): ((1, 1), (2, 0), (2, 1), (2, 2)),
    (2, 2): ((1, 2), (2, 1), (2, 2)),
}
_MASK_TO_SLOT = {frozenset(v): k for k, v in _MASKS.items()}
_LAYOUTS = ((3, 3, 3), (2, 3, 4), (2, 2, 2, 3), (5, 4), (1, 3, 5))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        _require(isinstance(value, (int, np.integer))
                 and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high,
                 f"{name} must be an integer in {low}..{high}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grids must have two dimensions in 1..30")
    _require(np.issubdtype(grid.dtype, np.integer)
             and bool(np.all((0 <= grid) & (grid <= 9))),
             "grid colors must be integers in 0..9")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.tobytes()


def _rectangles(grid, background):
    """Eight-connectivity enforces a background gap even at diagonals."""
    visited = grid == background
    rectangles = []
    height, width = grid.shape
    for start_r in range(height):
        for start_c in range(width):
            if visited[start_r, start_c]:
                continue
            visited[start_r, start_c] = True
            stack = [(start_r, start_c)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < height and 0 <= nc < width
                                and not visited[nr, nc]):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            bottom = max(r for r, _ in cells) + 1
            right = max(c for _, c in cells) + 1
            _require(len(cells) == (bottom - top) * (right - left),
                     "every object must be a filled rectangle")
            _require(min(bottom - top, right - left) >= 3,
                     "expanded 3x3 tiles must be at least 3x3")
            rectangles.append((top, left, bottom, right))
    _require(len(rectangles) == 9, "expected nine separated rectangles")
    return rectangles


def _collapse(tile):
    rows = np.r_[0, np.flatnonzero(np.any(tile[1:] != tile[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(tile[:, 1:] != tile[:, :-1], axis=0)) + 1]
    payload = tile[np.ix_(rows, cols)].copy()
    row_runs = tuple(map(int, np.diff(np.r_[rows, tile.shape[0]])))
    col_runs = tuple(map(int, np.diff(np.r_[cols, tile.shape[1]])))
    reconstructed = np.repeat(np.repeat(payload, row_runs, axis=0), col_runs, axis=1)
    _require(np.array_equal(reconstructed, tile), "invalid replication structure")
    return payload, row_runs, col_runs


def _assemble(tiles):
    output = np.empty((9, 9), dtype=np.int8)
    for (r, c), payload in tiles.items():
        output[3*r:3*r+3, 3*c:3*c+3] = payload
    return output


def _decode(input_grid):
    """All role inference, segmentation, normalization and routing is input-only."""
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    _require(all(bool(np.all(edge == background)) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "the complete outer border must show the background")
    rectangles = _rectangles(grid, background)
    common = set(range(10)) - {background}
    for top, left, bottom, right in rectangles:
        common.intersection_update(map(int, np.unique(grid[top:bottom, left:right])))
    _require(len(common) == 1, "expected one shared carrier color")
    carrier = common.pop()
    tiles, accents, records = {}, {}, []
    for box in rectangles:
        top, left, bottom, right = box
        crop = grid[top:bottom, left:right]
        collapsed, rows, cols = _collapse(crop)
        if collapsed.shape == (1, 1):
            _require(int(collapsed[0, 0]) == carrier, "plain tile must use carrier")
            slot = _CENTER
            payload = np.full((3, 3), carrier, dtype=np.int8)
        else:
            _require(collapsed.shape == (3, 3), "directional tile must collapse to 3x3")
            payload = collapsed
            other = payload != carrier
            values = np.unique(payload[other])
            _require(len(values) == 1, "a directional tile needs one uniform accent")
            mask = frozenset((int(r), int(c)) for r, c in np.argwhere(other))
            _require(mask in _MASK_TO_SLOT, "unrecognized corner or side accent")
            slot = _MASK_TO_SLOT[mask]
            accents[slot] = int(values[0])
        _require(slot not in tiles, "duplicate tile address")
        tiles[slot] = payload
        records.append({"slot": slot, "box": box, "rows": rows, "cols": cols,
                        "payload": payload, "crop": crop})
    _require(set(tiles) == set(_SLOTS), "missing tile address")
    return {"output": _assemble(tiles), "background": background, "carrier": carrier,
            "accents": accents, "records": records}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any input obeying the specified family, without generation state."""
    return _decode(input_grid)["output"]


def _tiles(carrier, accents):
    tiles = {_CENTER: np.full((3, 3), carrier, dtype=np.int8)}
    for slot, coordinates in _MASKS.items():
        payload = np.full((3, 3), carrier, dtype=np.int8)
        for cell in coordinates:
            payload[cell] = accents[slot]
        tiles[slot] = payload
    return tiles


def _grow(rng, minima, caps, amount):
    result = list(minima)
    for _ in range(amount):
        eligible = [i for i, (n, cap) in enumerate(zip(result, caps)) if n < cap]
        if not eligible:
            break
        result[int(rng.choice(eligible))] += 1
    return result


def _layout(rng, witness_size):
    """Guaranteed packing, followed by bounded independent rectangle moves."""
    counts = list(_LAYOUTS[int(rng.integers(len(_LAYOUTS)))])
    rng.shuffle(counts)
    shelf_ids = [s for s, n in enumerate(counts) for _ in range(n)]
    first_shelf, second_shelf = map(int, rng.choice(len(counts), 2, replace=False))
    first = int(rng.choice([i for i, s in enumerate(shelf_ids) if s == first_shelf]))
    second = int(rng.choice([i for i, s in enumerate(shelf_ids) if s == second_shelf]))
    forced = {first: witness_size, second: 6}
    min_heights = [max([3] + [n for i, n in forced.items() if shelf_ids[i] == s])
                   for s in range(len(counts))]
    min_widths = [forced.get(i, 3) for i in range(9)]
    needed_h = sum(min_heights) + len(counts) - 1
    needed_w = max(sum(min_widths[i] for i, s in enumerate(shelf_ids) if s == shelf)
                   + n - 1 for shelf, n in enumerate(counts))
    inner_h = int(rng.integers(needed_h, 29))
    inner_w = int(rng.integers(needed_w, 29))
    spare_h = inner_h - needed_h
    heights = _grow(rng, min_heights, [9] * len(counts),
                    int(rng.integers(spare_h + 1)))
    gaps_h = rng.multinomial(inner_h - sum(heights) - len(counts) + 1,
                             np.full(len(counts) + 1, 1 / (len(counts) + 1)))
    positions, shapes = [], []
    top = 1 + int(gaps_h[0])
    for shelf, count in enumerate(counts):
        indices = [i for i, s in enumerate(shelf_ids) if s == shelf]
        minima = [min_widths[i] for i in indices]
        spare_w = inner_w - sum(minima) - count + 1
        widths = _grow(rng, minima, [forced.get(i, 9) for i in indices],
                       int(rng.integers(spare_w + 1)))
        gaps_w = rng.multinomial(inner_w - sum(widths) - count + 1,
                                 np.full(count + 1, 1 / (count + 1)))
        left = 1 + int(gaps_w[0])
        for j, (index, width) in enumerate(zip(indices, widths)):
            height = forced.get(index)
            if height is None:
                height = int(rng.integers(3, heights[shelf] + 1))
            offset = int(rng.integers(heights[shelf] - height + 1))
            positions.append((top + offset, left))
            shapes.append((height, width))
            left += width + 1 + int(gaps_w[j + 1])
        top += heights[shelf] + 1 + int(gaps_h[shelf + 1])
    canvas_shape = (inner_h + 2, inner_w + 2)
    occupied = np.zeros(canvas_shape, dtype=bool)
    for (r, c), (h, w) in zip(positions, shapes):
        occupied[r:r+h, c:c+w] = True
    for _ in range(3):
        for index in rng.permutation(9):
            index = int(index)
            old_r, old_c = positions[index]
            h, w = shapes[index]
            occupied[old_r:old_r+h, old_c:old_c+w] = False
            chosen = (old_r, old_c)
            for _ in range(12):
                r = int(rng.integers(1, canvas_shape[0] - h))
                c = int(rng.integers(1, canvas_shape[1] - w))
                if not np.any(occupied[r-1:r+h+1, c-1:c+w+1]):
                    chosen = (r, c)
                    break
            positions[index] = chosen
            r, c = chosen
            occupied[r:r+h, c:c+w] = True
    if rng.integers(2):
        canvas_shape = canvas_shape[::-1]
        positions = [(c, r) for r, c in positions]
        shapes = [(w, h) for h, w in shapes]
    return canvas_shape, positions, shapes, first, second


def _profile(rng, total):
    choices = [(a, b, total-a-b) for a in range(1, 5) for b in range(1, 5)
               if 1 <= total-a-b <= 4]
    return choices[int(rng.integers(len(choices)))]


def _wrong_outputs(info):
    """Two concrete ablations: omit addressing, or replace collapse by resizing."""
    reading = _assemble({slot: record["payload"]
                         for slot, record in zip(_SLOTS, info["records"])})
    resized = {}
    for record in info["records"]:
        crop = record["crop"]
        rows = ((np.arange(3) + .5) * crop.shape[0] / 3).astype(int)
        cols = ((np.arange(3) + .5) * crop.shape[1] / 3).astype(int)
        resized[record["slot"]] = crop[np.ix_(rows, cols)]
    return reading, _assemble(resized)


def _make_pair(rng, roles, accent_count, test, shared_witness=False):
    background, carrier = roles
    witness_size = int(rng.integers(7, 10)) if test else 6
    canvas_shape, positions, shapes, first, second = _layout(rng, witness_size)
    corner = _CORNERS[int(rng.integers(4))]
    opposite = (2-corner[0], 2-corner[1])
    slots = [None] * 9
    slots[first], slots[second] = corner, opposite
    remaining = [s for s in _SLOTS if s not in (corner, opposite)]
    for index, slot in zip((i for i in range(9) if slots[i] is None),
                           [remaining[int(j)] for j in rng.permutation(7)]):
        slots[index] = slot
    colors = [int(c) for c in rng.permutation(10) if c not in roles][:accent_count]
    labels = list(range(accent_count)) + [int(rng.integers(accent_count))
                                         for _ in range(8-accent_count)]
    rng.shuffle(labels)
    accents = {slot: colors[label] for slot, label in zip(_DIRECTIONS, labels)}
    if shared_witness:
        # Preserve all selected colors while forcing opposing equal-color motifs.
        others = [s for s in _DIRECTIONS if s not in (corner, opposite)]
        rng.shuffle(others)
        accents = {corner: colors[0], opposite: colors[0]}
        for i, slot in enumerate(others):
            accents[slot] = colors[i+1] if i < accent_count-1 else int(rng.choice(colors))
    tiles = _tiles(carrier, accents)
    source = np.full(canvas_shape, background, dtype=np.int8)
    for index, (slot, (r, c), (h, w)) in enumerate(zip(slots, positions, shapes)):
        if index == first:
            rows = tuple(map(int, rng.permutation((1, 1, h-2))))
            cols = tuple(map(int, rng.permutation((1, 1, w-2))))
        elif index == second:
            rows = tuple(map(int, rng.permutation((1, 2, 3))))
            cols = tuple(map(int, rng.permutation((1, 2, 3))))
        else:
            rows, cols = _profile(rng, h), _profile(rng, w)
        expanded = np.repeat(np.repeat(tiles[slot], rows, axis=0), cols, axis=1)
        source[r:r+h, c:c+w] = expanded
    return {"input": source, "output": _assemble(tiles)}


def _pair_checks(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "each pair must have exactly input/output")
    source, target = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(source)
    _require(target.shape == (9, 9) and np.array_equal(target, info["output"]),
             "output disagrees with normalized motif assembly")
    _require(not np.any(target == info["background"]), "output contains background")
    for alternative in _wrong_outputs(info):
        _require(not np.array_equal(alternative, target), "shortcut is not distinguished")
    return source, info


def validate_generated_puzzle(puzzle):
    """Check input-derived invariants and the demonstration/shift guarantees."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_accent_counts = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            source, info = _pair_checks(pair)
            key = _key(source)
            _require(key not in seen, "duplicate input within puzzle")
            seen.add(key)
            directional = [r for r in info["records"] if r["slot"] != _CENTER]
            largest_run = max(max(r["rows"] + r["cols"]) for r in directional)
            if split == "train":
                train_accent_counts.add(len(set(info["accents"].values())))
                _require(largest_run == 4, "training must demonstrate bands through four")
            else:
                _require(5 <= largest_run <= 7, "test must extend a familiar uneven band")
    _require(8 in train_accent_counts and bool(train_accent_counts & {2, 3, 4}),
             "training must show distinct accents and substantial accent reuse")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_roles = bool(rng.integers(2))
    common_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    distinct_index, reuse_index = map(int, rng.permutation(num_train)[:2])
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == distinct_index:
                accent_count = 8
            elif split == "train" and index == reuse_index:
                accent_count = int(rng.integers(2, 5))
            else:
                accent_count = int(rng.integers(1, 7 if split == "test" else 9))
            roles = common_roles if stable_roles else tuple(map(int, rng.choice(10, 2, replace=False)))
            if not stable_roles and split == "train" and index == 1:
                first_background = int(puzzle["train"][0]["input"][0, 0])
                background = int(rng.choice([c for c in range(10) if c != first_background]))
                carrier = int(rng.choice([c for c in range(10) if c != background]))
                roles = (background, carrier)
            for _ in range(32):
                pair = _make_pair(rng, roles, accent_count, split == "test",
                                  split == "train" and index == reuse_index)
                info = _decode(pair["input"])
                wrong = _wrong_outputs(info)
                misplaced = sum(record["slot"] != slot
                                for slot, record in zip(_SLOTS, info["records"]))
                key = _key(pair["input"])
                if (key in used or misplaced < 6
                        or any(np.array_equal(item, pair["output"]) for item in wrong)):
                    continue
                used.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded search could not produce a distinct scrambled pair")
    validate_generated_puzzle(puzzle)
    return puzzle


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
    """42 streamed seeds; small aggregates only, no saved puzzles or traces."""
    signatures, shapes, profiles, palettes, carriers, backgrounds = (set() for _ in range(6))
    structures, center_shapes, stable_counts = set(), set(), [0, 0]
    cases = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        local_roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                backgrounds.add(info["background"])
                carriers.add(info["carrier"])
                local_roles.add((info["background"], info["carrier"]))
                palettes.add(len(set(info["accents"].values())))
                # Hash color-normalized input structure, without retaining full grids.
                renaming = {}
                normalized = bytes(renaming.setdefault(int(v), len(renaming))
                                   for v in pair["input"].flat)
                structures.add(hashlib.sha256(bytes(pair["input"].shape) + normalized).hexdigest())
                for record in info["records"]:
                    if record["slot"] == _CENTER:
                        center_shapes.add(record["crop"].shape)
                    else:
                        profiles.add((record["rows"], record["cols"]))
                cases += 1
        stable_counts[int(len(local_roles) > 1)] += 1
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 35 and len(profiles) >= 100
    assert backgrounds == carriers == set(range(10))
    assert palettes == set(range(1, 9))
    assert len(center_shapes) >= 10 and min(stable_counts) >= 5
    first = generate_puzzle(0)
    assert _digest(first) == _digest(generate_puzzle(0))
    pair = first["train"][0]
    permutation = np.array([7, 3, 8, 0, 5, 9, 1, 6, 2, 4], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert np.array_equal(solve(pair["input"].T), pair["output"].T)
    # All-unexpanded boundary member is valid for solve, outside the evidence schedule.
    tiles = _tiles(5, {s: 1 + i % 4 for i, s in enumerate(_DIRECTIONS)})
    fixture = np.zeros((13, 13), dtype=np.int8)
    for i, (r, c) in enumerate(_SLOTS):
        fixture[1+4*r:4+4*r, 1+4*c:4+4*c] = tiles[_SLOTS[(i+1) % 9]]
    assert np.array_equal(solve(fixture), _assemble(tiles))
    damaged = fixture.copy()
    damaged[1, 1] = 0
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("nonrectangular tile accepted")
    for counts in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(12345, *counts)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    return {"seeds": 42, "pairs": cases, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "color_normalized_inputs": len(structures),
            "directional_expansion_profiles": len(profiles), "accent_counts": sorted(palettes),
            "center_shapes": len(center_shapes), "stable_role_puzzles": stable_counts[0],
            "varying_role_puzzles": stable_counts[1]}
