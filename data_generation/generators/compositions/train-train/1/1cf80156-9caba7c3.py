"""Recover spaced marked stamps, compact their lattice, and outline their fusion.

Only standard-library modules and NumPy are used. No parent code or data is read.
"""

from __future__ import annotations

from functools import reduce
from math import gcd

import numpy as np


MARKER, OUTLINE, INTERIOR = 2, 7, 4
FIELD_COLORS = (0, 1, 3, 5, 6, 8, 9)
PERIMETER = ((0, 0), (0, 1), (0, 2), (1, 2),
             (2, 2), (2, 1), (2, 0), (1, 0))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must be greater than num_test")
    return a, b


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("grid must be a 1..30 integer rectangle with colors 0..9")
    return array.astype(np.int8, copy=False)


def _connected(mask):
    points = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not points:
        return False
    todo = [min(points)]
    visited = set(todo)
    while todo:
        r, c = todo.pop()
        for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if point in points and point not in visited:
                visited.add(point)
                todo.append(point)
    return visited == points


def _boundary(occupied):
    padded = np.pad(occupied, 1, constant_values=False)
    surrounded = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                  & padded[1:-1, :-2] & padded[1:-1, 2:])
    return occupied & ~surrounded


def _holes(mask):
    empty = ~np.pad(mask, 1, constant_values=False)
    unseen = {tuple(map(int, p)) for p in np.argwhere(empty)}
    components = 0
    while unseen:
        components += 1
        start = min(unseen)
        unseen.remove(start)
        todo = [start]
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
    return components - 1


def _candidates(grid, carrier):
    windows = np.lib.stride_tricks.sliding_window_view(grid, (3, 3))
    full = np.all((windows == carrier) | (windows == MARKER), axis=(-2, -1))
    numbers = np.sum(windows == MARKER, axis=(-2, -1))
    valid = full & (windows[:, :, 1, 1] == carrier) & (numbers >= 1) & (numbers <= 6)
    width = grid.shape[1]
    result = []
    for r0, c0 in np.argwhere(valid):
        r, c = int(r0), int(c0)
        cells = sum(1 << ((r + dr) * width + c + dc)
                    for dr in range(3) for dc in range(3))
        marks = sum(1 << ((r + dr) * width + c + dc)
                    for dr, dc in PERIMETER if grid[r + dr, c + dc] == MARKER)
        result.append((r, c, cells, marks))
    return result, int(full.sum()), int(np.count_nonzero(full & (numbers == 0)))


def _unique_cover(grid, carrier):
    """Enumerate marker covers, stopping after the second distinct cover."""
    candidates, full_count, carrier_only = _candidates(grid, carrier)
    width = grid.shape[1]
    marker_indices = [int(r) * width + int(c) for r, c in np.argwhere(grid == MARKER)]
    all_marks = sum(1 << i for i in marker_indices)
    by_marker = {}
    for i in marker_indices:
        bit = 1 << i
        by_marker[bit] = [j for j, candidate in enumerate(candidates) if candidate[3] & bit]
    answers = []

    def visit(remaining, used, chosen):
        if len(answers) == 2:
            return
        if remaining == 0:
            answers.append(tuple(sorted(chosen)))
            return
        choices = None
        bits = remaining
        while bits:
            bit = bits & -bits
            options = [j for j in by_marker[bit]
                       if not candidates[j][2] & used
                       and candidates[j][3] & remaining == candidates[j][3]]
            if not options:
                return
            if choices is None or len(options) < len(choices):
                choices = options
            bits ^= bit
        for j in choices:
            r, c, footprint, marked = candidates[j]
            visit(remaining ^ marked, used | footprint, chosen + ((r, c),))
            if len(answers) == 2:
                break

    if all_marks:
        visit(all_marks, 0, ())
    if len(answers) != 1:
        raise ValueError("input must have exactly one nonoverlapping marker cover")
    return answers[0], len(candidates), full_count, carrier_only


def _decode(value):
    grid = _grid(value)
    if min(grid.shape) < 3:
        raise ValueError("no complete stamps")
    corners = [int(grid[r, c]) for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    if len(set(corners)) != 1 or corners[0] not in FIELD_COLORS:
        raise ValueError("four corners must show the background")
    background = corners[0]
    colors = set(map(int, np.unique(grid)))
    other = colors - {background, MARKER}
    if len(colors) != 3 or MARKER not in colors or len(other) != 1:
        raise ValueError("input must have background, carrier, and marker")
    carrier = other.pop()
    if carrier not in FIELD_COLORS:
        raise ValueError("invalid carrier")
    positions, candidate_count, full_count, carrier_only = _unique_cover(grid, carrier)
    min_r = min(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    pitch_r = reduce(gcd, (r - min_r for r, _ in positions))
    pitch_c = reduce(gcd, (c - min_c for _, c in positions))
    if not (4 <= pitch_r <= 7 and 4 <= pitch_c <= 7):
        raise ValueError("both lattice pitches must be visibly determined in 4..7")
    h = (max(r for r, _ in positions) - min_r) // pitch_r + 1
    w = (max(c for _, c in positions) - min_c) // pitch_c + 1
    if not (2 <= h <= 6 and 2 <= w <= 6):
        raise ValueError("coarse extents must be 2..6")
    coarse = np.zeros((h, w), dtype=bool)
    masks = {}
    for r, c in positions:
        row, col = (r - min_r) // pitch_r, (c - min_c) // pitch_c
        coarse[row, col] = True
        masks[row, col] = sum(1 << i for i, (dr, dc) in enumerate(PERIMETER)
                              if grid[r + dr, c + dc] == MARKER)
    if not _connected(coarse):
        raise ValueError("decoded mosaic must be four-connected")
    decoys = 0
    for row, col in np.argwhere(~coarse):
        r, c = min_r + int(row) * pitch_r, min_c + int(col) * pitch_c
        decoys += int(np.all(grid[r:r + 3, c:c + 3] == carrier))
    return {"background": background, "carrier": carrier,
            "positions": positions, "pitch": (pitch_r, pitch_c),
            "origin": (min_r, min_c), "coarse": coarse, "masks": masks,
            "candidates": candidate_count, "full": full_count,
            "carrier_only": carrier_only, "missing_site_decoys": decoys}


def _render(coarse, masks, background, fused=True):
    occupied = np.repeat(np.repeat(coarse, 3, axis=0), 3, axis=1)
    output = np.full(occupied.shape, background, dtype=np.int8)
    output[occupied] = INTERIOR
    if fused:
        output[_boundary(occupied)] = OUTLINE
    else:
        for row, col in masks:
            for dr, dc in PERIMETER:
                output[3 * row + dr, 3 * col + dc] = OUTLINE
    for (row, col), mask in masks.items():
        for bit, (dr, dc) in enumerate(PERIMETER):
            if mask & (1 << bit):
                output[3 * row + dr, 3 * col + dc] = MARKER
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input pixels alone; no generator parameters are consulted."""
    info = _decode(input_grid)
    return _render(info["coarse"], info["masks"], info["background"])


def _path(rng, h, w):
    result = np.zeros((h, w), dtype=bool)
    steps = [0] * (h - 1) + [1] * (w - 1)
    rng.shuffle(steps)
    row = col = 0
    result[row, col] = True
    for step in steps:
        row += step == 0
        col += step == 1
        result[row, col] = True
    return result


def _shape(rng, h, w, mode):
    if mode == "ring":
        result = np.zeros((h, w), dtype=bool)
        result[[0, -1], :] = True
        result[:, [0, -1]] = True
    elif mode == "hole":
        result = np.ones((h, w), dtype=bool)
        result[int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))] = False
    else:
        result = _path(rng, h, w)
        if mode == "growth":
            goal = int(rng.integers(int(result.sum()), h * w))
            while int(result.sum()) < goal:
                frontier = [(r, c) for r in range(h) for c in range(w)
                            if not result[r, c] and any(
                                0 <= rr < h and 0 <= cc < w and result[rr, cc]
                                for rr, cc in ((r - 1, c), (r + 1, c),
                                               (r, c - 1), (r, c + 1)))]
                result[frontier[int(rng.integers(len(frontier)))]] = True
    if rng.integers(2):
        result = result[::-1, :]
    if rng.integers(2):
        result = result[:, ::-1]
    return result.copy()


def _layout(rng, split, index, first):
    if split == "train":
        h, w = map(int, rng.integers(2, 5, size=2))
        if index == 0:
            mode = "path"
        elif index == 1:
            h, w = map(int, rng.integers(3, 5, size=2))
            if (h, w) == first["coarse"].shape:
                h = 7 - h  # interchange 3 and 4
            mode = "ring"
        else:
            modes = ["path", "growth"] + (["ring", "hole"] if min(h, w) >= 3 else [])
            mode = modes[int(rng.integers(len(modes)))]
    else:
        h, w = int(rng.integers(5, 7)), int(rng.integers(2, 6))
        if rng.integers(2):
            h, w = w, h
        modes = ["path", "growth"] + (["ring", "hole"] if min(h, w) >= 3 else [])
        mode = modes[int(rng.integers(len(modes)))]
    coarse = _shape(rng, h, w, mode)
    pitch = tuple(int(rng.integers(4, min(7, 27 // (n - 1)) + 1)) for n in (h, w))
    if split == "train" and index == 1:
        options = [(a, b) for a in range(4, 8) for b in range(4, 8)
                   if a != b and (a, b) != first["pitch"]]
        pitch = options[int(rng.integers(len(options)))]
    return coarse, pitch


def _make_pair(rng, split, index, first):
    coarse, pitch = _layout(rng, split, index, first)
    h, w = coarse.shape
    span_r, span_c = 3 + (h - 1) * pitch[0], 3 + (w - 1) * pitch[1]
    background, carrier = map(int, rng.choice(FIELD_COLORS, size=2, replace=False))
    if split == "train" and index == 1 and background == first["background"]:
        background = int(rng.choice([c for c in FIELD_COLORS
                                     if c not in (carrier, first["background"])]))
    rich = split == "test" or index == 1 or (index > 1 and bool(rng.integers(2)))
    sites = [tuple(map(int, p)) for p in np.argwhere(coarse)]
    empty_sites = [tuple(map(int, p)) for p in np.argwhere(~coarse)]
    for _ in range(500):
        extra_r = int(rng.integers(0, 31 - span_r))
        extra_c = int(rng.integers(0, 31 - span_c))
        top, left = int(rng.integers(extra_r + 1)), int(rng.integers(extra_c + 1))
        shape = (span_r + extra_r, span_c + extra_c)
        positions = [(top + row * pitch[0], left + col * pitch[1]) for row, col in sites]
        protected = np.zeros(shape, dtype=bool)
        for r, c in positions:
            protected[r:r + 3, c:c + 3] = True
        corners = ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0),
                   (shape[0] - 1, shape[1] - 1))
        if any(protected[corner] for corner in corners):
            continue
        density = float(rng.uniform(.49, .67) if rich else rng.uniform(.29, .49))
        grid = np.where(rng.random(shape) < density, carrier, background).astype(np.int8)
        masks = {}
        for (row, col), (r, c) in zip(sites, positions):
            grid[r:r + 3, c:c + 3] = carrier
            # The first demonstration exposes full-tile extent beyond marker extrema.
            eligible = [i for i, (dr, _) in enumerate(PERIMETER)
                        if not (split == "train" and index == 0 and row == 0 and dr == 0)]
            number = int(rng.integers(1, min(6, len(eligible)) + 1))
            mask = sum(1 << int(i) for i in rng.choice(eligible, size=number, replace=False))
            masks[row, col] = mask
            for bit, (dr, dc) in enumerate(PERIMETER):
                if mask & (1 << bit):
                    grid[r + dr, c + dc] = MARKER
        # A complete but unmarked tile at a genuine missing lattice site.
        row, col = empty_sites[int(rng.integers(len(empty_sites)))]
        r, c = top + row * pitch[0], left + col * pitch[1]
        if any(r <= rr < r + 3 and c <= cc < c + 3 for rr, cc in corners):
            continue
        grid[r:r + 3, c:c + 3] = carrier
        for corner in corners:
            grid[corner] = background
        try:
            info = _decode(grid)
        except ValueError:
            continue
        if set(info["positions"]) != set(positions) or info["pitch"] != pitch:
            continue
        if rich and info["candidates"] <= len(positions):
            continue
        output = _render(coarse, masks, background)
        if np.array_equal(output, _render(coarse, masks, background, fused=False)):
            continue
        return {"input": grid, "output": output}, info
    raise RuntimeError("could not construct uniquely decodable stamps")


def _key(array):
    array = np.asarray(array, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    first = None
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            for _ in range(30):
                pair, info = _make_pair(rng, split, index, first)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
            if first is None:
                first = info
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(grid)
    coarse, masks, bg = info["coarse"], info["masks"], info["background"]
    expected = _render(coarse, masks, bg)
    assert np.array_equal(output, expected), "incorrect fused mosaic"
    # Check transported markers and background slots directly, not just dimensions.
    for (row, col), mask in masks.items():
        for bit, (dr, dc) in enumerate(PERIMETER):
            assert (output[3 * row + dr, 3 * col + dc] == MARKER) == bool(mask & (1 << bit))
        assert output[3 * row + 1, 3 * col + 1] == INTERIOR
    for row, col in np.argwhere(~coarse):
        assert np.all(output[3 * row:3 * row + 3, 3 * col:3 * col + 3] == bg)
    assert output.shape == tuple(3 * n for n in coarse.shape)
    assert np.any(output == OUTLINE) and np.any(output == INTERIOR)
    info["seam_witness"] = int(np.count_nonzero(
        output != _render(coarse, masks, bg, fused=False)))
    info["holes"] = _holes(coarse)
    info["edge"] = any(r == 0 or c == 0 or r + 3 == grid.shape[0]
                       or c + 3 == grid.shape[1] for r, c in info["positions"])
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(info)
            # This validator checks the generated sampling regime as well as the rule.
            assert np.any(~info["coarse"]), "generated pair needs a gap witness"
            assert info["missing_site_decoys"] >= 1, "missing marker-use witness"
            assert info["seam_witness"] > 0, "fusion has no visible effect"
    train, test = analyses["train"], analyses["test"]
    assert train[0]["holes"] == 0 and train[1]["holes"] >= 1
    assert train[0]["coarse"].shape != train[1]["coarse"].shape
    assert train[0]["pitch"] != train[1]["pitch"]
    assert train[1]["pitch"][0] != train[1]["pitch"][1]
    assert train[0]["background"] != train[1]["background"]
    markers = np.argwhere(np.asarray(puzzle["train"][0]["input"]) == MARKER)
    assert int(markers[:, 0].min()) > train[0]["origin"][0]
    assert train[1]["candidates"] > len(train[1]["positions"])
    assert all(max(i["coarse"].shape) <= 4 for i in train)
    assert all(5 <= max(i["coarse"].shape) <= 6 for i in test)
    assert all(i["candidates"] > len(i["positions"]) for i in test)
    # Removing compaction (A's normalized coordinate frame) changes both extents.
    for i in train:
        spaced = tuple(3 + (n - 1) * p for n, p in zip(i["coarse"].shape, i["pitch"]))
        assert all(a > b for a, b in zip(spaced, (3 * n for n in i["coarse"].shape)))


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity and family-invariant suite; return measurements."""
    assert _same(generate_puzzle(0), generate_puzzle(0))
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        puzzle = generate_puzzle(7200 + a * 10 + b, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                 (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    inputs, outputs, occupancies, pitches, colors, counts, masks, signatures = [set() for _ in range(8)]
    holes = edges = competing = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                info = _inspect_pair(pair)
                inputs.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                occupancies.add(_key(info["coarse"]))
                pitches.add(info["pitch"])
                colors.add((info["background"], info["carrier"]))
                counts.add(len(info["positions"]))
                masks.update(mask.bit_count() for mask in info["masks"].values())
                holes += info["holes"] > 0
                edges += info["edge"]
                competing += info["candidates"] > len(info["positions"])
    assert len(signatures) == 42
    assert len(inputs) >= 35 and len(outputs) >= 15
    assert len(occupancies) >= 65 and len(pitches) >= 12
    assert len(colors) >= 30 and len(counts) >= 10
    assert masks == set(range(1, 7)) and holes >= 42 and edges >= 5 and competing >= 84
    # Valid rule members outside the teaching sampler: no gap, maximum extent,
    # dense marker masks, and independently selected pitch. The public solver
    # must not require the sampler's missing-site decoy or sparse first shape.
    for shape, pitch, bg, carrier in (((3, 4), (4, 6), 0, 9),
                                      ((6, 6), (4, 4), 8, 0)):
        coarse = np.ones(shape, dtype=bool)
        masks_full = {(r, c): 63 for r in range(shape[0]) for c in range(shape[1])}
        canvas = np.full(tuple(5 + (n - 1) * p for n, p in zip(shape, pitch)),
                         bg, dtype=np.int8)
        for row, col in masks_full:
            r, c = 1 + row * pitch[0], 1 + col * pitch[1]
            canvas[r:r + 3, c:c + 3] = carrier
            for bit, (dr, dc) in enumerate(PERIMETER):
                if masks_full[row, col] & (1 << bit):
                    canvas[r + dr, c + dc] = MARKER
        assert np.array_equal(solve(canvas), _render(coarse, masks_full, bg))
    # The validator must actually reject a wrong boundary pixel.
    damaged = generate_puzzle(123, 2, 1)
    out = damaged["train"][0]["output"]
    r, c = map(int, np.argwhere(out == OUTLINE)[0])
    out[r, c] = INTERIOR
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged outline")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(inputs),
            "output_shapes": len(outputs), "coarse_shapes": len(occupancies),
            "pitch_pairs": len(pitches), "palette_roles": len(colors),
            "stamp_counts": sorted(counts), "marker_counts": sorted(masks),
            "pairs_with_holes": holes, "pairs_with_edge_stamps": edges,
            "pairs_with_competing_candidates": competing}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
