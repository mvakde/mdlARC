"""Mirrored inks, unreflected stencils: a two-scale erased tapestry.

Only the coarse ink map reflects. Every tile prints the SAME oriented binary
stencil. Thin rails encode the lattice; a unique rectangular color is erased.
The answer is the erased crop, including reconstructed rails and carrier gaps.
All inference is input-only. No parent code, files, seed, or training data is used.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0, 140277108, 247107321, 253349088, 462312490, 138579876,
              274541214, 184002731, 270871599, 464999315, 374026452,
              570389184, 284306370, 695383073, 457099879, 846351983,
              193237449, 993297353, 613148111, 18268495, 962670039,
              648625130, 987598201, 159170585, 29531245, 563458803,
              413276454, 907757522, 406637372, 185838568, 825251597,
              525607308, 879878718, 866175693, 292497728, 835486776,
              90636664, 23535856, 599883076, 694258056, 309078149, 931865890)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "Grid must have two dimensions of size 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "Grid cells must be integers in 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _full_box(mask):
    return (0 < int(mask.sum()) < mask.size and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


def _local_agreements(mask):
    """Coordinate maps for plausible local rotations/reflections/cyclic phases.

    Deduplicate maps: on two-cell axes, a flip and a cyclic shift can coincide.
    This is teaching coverage, never a convention used by solve().
    """
    ids = np.arange(mask.size).reshape(mask.shape)
    views = (ids, ids[::-1], ids[:, ::-1], ids[::-1, ::-1],
             ids.T, ids.T[::-1], ids.T[:, ::-1], ids.T[::-1, ::-1])
    maps = set()
    for view in views:
        if view.shape == mask.shape:
            for dr in range(mask.shape[0]):
                for dc in range(mask.shape[1]):
                    maps.add(tuple(int(x) for x in np.roll(view, (dr, dc), (0, 1)).flat))
    flat = mask.ravel()
    return {mapping for mapping in maps if np.array_equal(flat[list(mapping)], flat)}


def _paint(inks, stencil, rail, carrier):
    """Construction renderer: stamp each original oriented tile independently."""
    m, n = inks.shape
    p, q = stencil.shape
    canvas = np.full((m * (p + 1) + 1, n * (q + 1) + 1), rail, dtype=np.int8)
    for i in range(m):
        for j in range(n):
            r, c = 1 + i * (p + 1), 1 + j * (q + 1)
            canvas[r:r + p, c:c + q] = np.where(stencil, inks[i, j], carrier)
    return canvas


def _decode(input_grid):
    """Factor pixels into a periodic stencil and a reflected coarse ink map."""
    a = _grid(input_grid)
    rail = int(a[0, 0])
    _require(np.all(a[0] == rail) and np.all(a[-1] == rail)
             and np.all(a[:, 0] == rail) and np.all(a[:, -1] == rail),
             "The outside rail must be intact")
    rectangles = []
    for color in np.unique(a):
        if color == rail:
            continue
        rr, cc = np.nonzero(a == color)
        r, s = int(rr.min()), int(rr.max()) + 1
        c, d = int(cc.min()), int(cc.max()) + 1
        if (s - r) * (d - c) == len(rr) and len(rr) >= 2:
            rectangles.append((int(color), (r, s, c, d)))
    _require(len(rectangles) == 1, "Erasure must be the unique rectangular color")
    eraser, box = rectangles[0]
    visible = a != eraser
    row_rails = np.flatnonzero(np.all((a == rail) | ~visible, axis=1))
    col_rails = np.flatnonzero(np.all((a == rail) | ~visible, axis=0))
    _require(len(row_rails) >= 5 and len(col_rails) >= 5, "Lattice needs 4+ tiles per axis")
    dp, dq = np.diff(row_rails), np.diff(col_rails)
    _require(np.all(dp == dp[0]) and np.all(dq == dq[0]), "Rails must be equally spaced")
    p, q = int(dp[0]) - 1, int(dq[0]) - 1
    m, n = len(row_rails) - 1, len(col_rails) - 1
    _require(2 <= p <= 4 and 2 <= q <= 4, "Tile sides must be 2..4")
    _require(a.shape == (m * (p + 1) + 1, n * (q + 1) + 1), "Incomplete lattice")
    rr, cc = np.indices(a.shape)
    rail_cells = (rr % (p + 1) == 0) | (cc % (q + 1) == 0)
    _require(np.all(a[visible & rail_cells] == rail)
             and np.all(a[visible & ~rail_cells] != rail), "Rail color inside tile")
    # Sample every local coordinate across the lattice. -1 denotes erased data.
    tiles = np.empty((m, n, p, q), dtype=np.int16)
    for u in range(p):
        for v in range(q):
            samples = a[1 + u:-1:p + 1, 1 + v:-1:q + 1]
            tiles[:, :, u, v] = np.where(samples == eraser, -1, samples)
    candidates = []
    for carrier in (int(x) for x in np.unique(a) if x not in (rail, eraser)):
        stencil = np.zeros((p, q), dtype=bool)
        valid = True
        for u in range(p):
            for v in range(q):
                seen = tiles[:, :, u, v]
                seen = seen[seen >= 0]
                if not seen.size or (np.any(seen == carrier) and np.any(seen != carrier)):
                    valid = False
                    break
                stencil[u, v] = bool(np.all(seen != carrier))
            if not valid:
                break
        if not valid or not _full_box(stencil):
            continue
        known = np.full((m, n), -1, dtype=np.int16)
        for i in range(m):
            for j in range(n):
                colors = np.unique(tiles[i, j][stencil])
                colors = colors[colors >= 0]
                if len(colors) > 1 or carrier in colors:
                    valid = False
                    break
                if len(colors):
                    known[i, j] = colors[0]
            if not valid:
                break
        if not valid:
            continue
        inks = np.full((m, n), -1, dtype=np.int16)
        for i in range((m + 1) // 2):
            for j in range((n + 1) // 2):
                orbit = {(i, j), (m - 1 - i, j), (i, n - 1 - j), (m - 1 - i, n - 1 - j)}
                colors = {int(known[y, x]) for y, x in orbit if known[y, x] >= 0}
                if len(colors) != 1:
                    valid = False
                    break
                value = next(iter(colors))
                for y, x in orbit:
                    inks[y, x] = value
            if not valid:
                break
        if not valid or len(np.unique(inks)) < 2:
            continue
        # Separate from stamp-based generation: decode all canvas coordinates.
        full = np.full(a.shape, rail, dtype=np.int8)
        rows, cols = np.nonzero(~rail_cells)
        local_r, local_c = rows % (p + 1) - 1, cols % (q + 1) - 1
        occupied = stencil[local_r, local_c]
        full[rows, cols] = np.where(occupied, inks[rows // (p + 1), cols // (q + 1)], carrier)
        if not np.array_equal(full[visible], a[visible]):
            continue
        r, s, c, d = box
        candidates.append({"rail": rail, "carrier": carrier, "eraser": eraser,
                           "box": box, "tile_shape": (p, q), "lattice_shape": (m, n),
                           "stencil": stencil, "known": known, "inks": inks.astype(np.int8),
                           "rail_cells": rail_cells, "full": full,
                           "answer": full[r:s, c:d].copy()})
    _require(len(candidates) == 1, "No unique stencil/carrier and mirrored-ink reconstruction")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the erased rectangle; receives no generation conventions."""
    return _decode(input_grid)["answer"]


def _checks(a, record):
    """Measure actual witnesses and wrong-rule answers from the returned pixels."""
    r, s, c, d = record["box"]
    p, q = record["tile_shape"]
    m, n = record["lattice_shape"]
    eraser, carrier, rail = (record[k] for k in ("eraser", "carrier", "rail"))
    out = record["answer"]
    target_inks = set(int(x) for x in np.unique(out)) - {rail, carrier}
    hidden_tiles = int(np.count_nonzero(record["known"] < 0))
    literal_trials = literal_errors = 0
    for code in (1, 2, 3):
        # Query each missing cell's whole-canvas mirror, with orientation restored
        # to the query. This is precisely the tempting wrong pixel-scale rule.
        source = a[::-1] if code == 1 else a[:, ::-1] if code == 2 else a[::-1, ::-1]
        guess = source[r:s, c:d]
        if not np.any(guess == eraser):
            literal_trials += 1
            literal_errors += not np.array_equal(guess, out)
    # Check either axis on the coarse map, not on the full pixel image.
    known = record["known"]
    needs_other_axis = False
    for i, j in np.argwhere(known < 0):
        needs_other_axis |= known[m - 1 - i, j] < 0 or known[i, n - 1 - j] < 0
    aligned = ((r - 1) % (p + 1) == 0 and s % (p + 1) == 0
               and (c - 1) % (q + 1) == 0 and d % (q + 1) == 0)
    return {"target_ink_count": len(target_inks), "hidden_tiles": hidden_tiles,
            "literal_trials": literal_trials, "literal_errors": literal_errors,
            "needs_other_axis": bool(needs_other_axis), "aligned": aligned,
            "has_carrier": bool(np.any(out == carrier)), "has_rails": bool(np.any(out == rail))}


def _stencil(rng, p, q):
    for _ in range(128):
        mask = np.zeros(p * q, dtype=bool)
        count = int(rng.integers(2, p * q))
        mask[rng.choice(p * q, count, replace=False)] = True
        mask = mask.reshape(p, q)
        if (_full_box(mask) and all(not np.array_equal(mask, other)
                                    for other in (mask[::-1], mask[:, ::-1], mask[::-1, ::-1]))):
            return mask
    # Guaranteed finite fallback, still orientation-sensitive on both axes.
    mask = np.ones((p, q), dtype=bool)
    mask[0, 0] = False
    return np.rot90(mask, 2 * int(rng.integers(2))).copy()


def _window(rng, m, n, p, q, seam, partial):
    # Work in tile coordinates first. Staying in one strict half of the
    # other axis guarantees a surviving member of every coarse ink orbit.
    if not seam:
        h = int(rng.integers(1, m // 2 + 1))
        w = int(rng.integers(1, n // 2 + 1))
        if h * w == 1:
            w = 2
        top = int(rng.integers(0, m // 2 - h + 1))
        left = int(rng.integers(0, n // 2 - w + 1))
    elif int(rng.integers(2)) == 0:
        h = int(rng.integers(3, min(m, 5) + 1)) if partial else 2 + m % 2
        low, high = max(0, (m + 1) // 2 - h), min(m // 2 - 1, m - h)
        top = int(rng.integers(low, high + 1))
        w = int(rng.integers(2, n // 2 + 1))
        left = int(rng.integers(0, n // 2 - w + 1))
    else:
        w = int(rng.integers(3, min(n, 5) + 1)) if partial else 2 + n % 2
        low, high = max(0, (n + 1) // 2 - w), min(n // 2 - 1, n - w)
        left = int(rng.integers(low, high + 1))
        h = int(rng.integers(2, m // 2 + 1))
        top = int(rng.integers(0, m // 2 - h + 1))
    if rng.integers(2):
        top = m - top - h
    if rng.integers(2):
        left = n - left - w
    r, s = 1 + top * (p + 1), (top + h) * (p + 1)
    c, d = 1 + left * (q + 1), (left + w) * (q + 1)
    if partial:
        # Crop one boundary on each axis; opposite edge tiles remain complete.
        if rng.integers(2):
            r += int(rng.integers(1, p))
        else:
            s -= int(rng.integers(1, p))
        if rng.integers(2):
            c += int(rng.integers(1, q))
        else:
            d -= int(rng.integers(1, q))
    return r, s, c, d


def _make_pair(rng, tile_shape, palette, seam, partial, used, witness=None):
    p, q = tile_shape
    rail, carrier, eraser, *ink_palette = palette
    surviving = _local_agreements(witness) if witness is not None else None
    identity = tuple(range(p * q))
    for _ in range(96):
        m = int(rng.integers(4, min(8, 29 // (p + 1)) + 1))
        n = int(rng.integers(4, min(8, 29 // (q + 1)) + 1))
        base = rng.choice(ink_palette, ((m + 1) // 2, (n + 1) // 2))
        # Each chosen ink occurs in a visible orbit; no literal ink is privileged.
        base.flat[:len(ink_palette)] = rng.permutation(ink_palette)
        rows = np.minimum(np.arange(m), m - 1 - np.arange(m))
        cols = np.minimum(np.arange(n), n - 1 - np.arange(n))
        inks = base[rows[:, None], cols[None, :]].astype(np.int8)
        stencil = _stencil(rng, p, q)
        if witness is not None and (np.array_equal(stencil, witness)
                                    or surviving & _local_agreements(stencil) != {identity}):
            continue
        full = _paint(inks, stencil, rail, carrier)
        r, s, c, d = _window(rng, m, n, p, q, seam, partial)
        a = full.copy()
        a[r:s, c:d] = eraser
        if _key(a) in used:
            continue
        record = _decode(a)
        facts = _checks(a, record)
        if (facts["target_ink_count"] < 2 or not facts["hidden_tiles"]
                or not facts["literal_trials"] or facts["literal_errors"] != facts["literal_trials"]
                or not facts["has_carrier"] or not facts["has_rails"]
                or (seam and not facts["needs_other_axis"])):
            continue
        output = full[r:s, c:d].copy()
        _require(np.array_equal(output, record["answer"]), "Construction/inference mismatch")
        return {"input": a, "output": output}
    raise RuntimeError("Bounded tapestry sampling exhausted")


def _counts(num_train, num_test):
    for count, name, lower, upper in ((num_train, "num_train", 2, 10),
                                       (num_test, "num_test", 1, 4)):
        _require(not isinstance(count, (bool, np.bool_)) and isinstance(count, (int, np.integer))
                 and lower <= int(count) <= upper, name + " is not a legal integer count")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be an integer")
    rng = np.random.default_rng(seed)
    tile_shape = tuple(int(x) for x in rng.integers(2, 5, size=2))
    # At least four independent coarse orbits exist even in a 4x4 lattice.
    ink_count = int(rng.integers(2, 5))
    palette = [int(x) for x in rng.permutation(10)[:3 + ink_count]]
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            partial = split == "test"
            seam = partial or index == 1 or (index > 1 and bool(rng.integers(2)))
            witness = _decode(puzzle["train"][0]["input"])["stencil"] if split == "train" and index == 1 else None
            pair = _make_pair(rng, tile_shape, palette, seam, partial, used, witness)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantics from arrays, including omission and teaching witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, conventions = set(), set()
    local_orders, training_stencils = None, set()
    facts_by_split = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in used, "Duplicate input")
            used.add(_key(a))
            record = _decode(a)
            _require(np.array_equal(b, record["answer"]), "Wrong erased crop")
            _require(record["eraser"] not in b, "Eraser leaked into the answer")
            _require(np.array_equal(record["inks"], record["inks"][::-1])
                     and np.array_equal(record["inks"], record["inks"][:, ::-1]), "Coarse reflection broken")
            facts = _checks(a, record)
            _require(facts["hidden_tiles"] > 0, "No genuinely missing tile ink")
            _require(facts["target_ink_count"] >= 2, "One-ink periodic shortcut is not distinguished")
            _require(facts["has_carrier"] and facts["has_rails"], "Answer lacks stencil gaps or rail evidence")
            _require(facts["literal_trials"] > 0 and facts["literal_errors"] == facts["literal_trials"],
                     "Literal pixel reflection shortcut is not distinguished")
            facts_by_split[split].append(facts)
            if split == "train":
                orders = _local_agreements(record["stencil"])
                local_orders = orders if local_orders is None else local_orders & orders
                training_stencils.add(record["stencil"].tobytes())
            conventions.add((record["tile_shape"], record["rail"], record["carrier"], record["eraser"],
                             tuple(int(x) for x in np.unique(record["inks"]))))
    _require(len(conventions) == 1, "Puzzle conventions vary")
    tile_shape = next(iter(conventions))[0]
    _require(local_orders == {tuple(range(tile_shape[0] * tile_shape[1]))}
             and len(training_stencils) >= 2, "Training leaves a local orientation or cyclic phase ambiguity")
    train, test = facts_by_split["train"], facts_by_split["test"]
    _require(all(f["aligned"] for f in train), "Generated training windows must show complete tiles")
    _require(any(not f["needs_other_axis"] for f in train)
             and any(f["needs_other_axis"] for f in train), "Missing direct/seam teaching witnesses")
    _require(all(not f["aligned"] and f["needs_other_axis"] for f in test), "Missing partial-tile test shift")
    return True


def variation_report(seeds=HOST_SEEDS):
    """Bounded aggregate only: retain hashes/features, never all sampled grids."""
    puzzle_hashes, input_hashes, output_hashes = set(), set(), set()
    tile_shapes, lattice_shapes, input_shapes, output_shapes = set(), set(), set(), set()
    masks, stencils, densities, palettes, roles = set(), set(), set(), set(), [set() for _ in range(3)]
    pairs = hidden = seam = partial = pixel_wrong = 0
    for seed in seeds:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                rec = _decode(a)
                facts = _checks(a, rec)
                pairs += 1
                hidden += facts["hidden_tiles"]
                seam += facts["needs_other_axis"]
                partial += not facts["aligned"]
                pixel_wrong += facts["literal_errors"]
                for arr, dest in ((a, input_hashes), (b, output_hashes)):
                    dest.add(hashlib.sha256(bytes(arr.shape) + arr.tobytes()).hexdigest())
                tile_shapes.add(rec["tile_shape"])
                lattice_shapes.add(rec["lattice_shape"])
                input_shapes.add(a.shape)
                output_shapes.add(b.shape)
                masks.add(rec["box"])
                stencils.add((rec["tile_shape"], rec["stencil"].tobytes()))
                densities.add((rec["stencil"].size, int(rec["stencil"].sum())))
                palettes.add(tuple(int(x) for x in np.unique(rec["inks"])))
                for dest, role in zip(roles, ("rail", "carrier", "eraser")):
                    dest.add(rec[role])
    return {"seeds": len(seeds), "pairs": pairs, "distinct_puzzles": len(puzzle_hashes),
            "distinct_inputs": len(input_hashes), "distinct_outputs": len(output_hashes),
            "tile_shapes": sorted(tile_shapes), "lattice_shape_count": len(lattice_shapes),
            "input_shape_count": len(input_shapes), "output_shape_count": len(output_shapes),
            "window_count": len(masks), "stencil_count": len(stencils),
            "area_occupancy_combinations": sorted(densities), "ink_palette_count": len(palettes),
            "role_colors": [sorted(x) for x in roles], "fully_ink_erased_tiles": hidden,
            "seam_cases": seam, "partial_tile_cases": partial, "wrong_pixel_mirror_predictions": pixel_wrong,
            "semantic_verification": "generator-owned checks; no independent or blind solver"}


def _expect_failure(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Invalid or corrupted example was accepted")


def self_test():
    """42 provided seeds, count boundaries, determinism and semantic mutations."""
    report = variation_report()
    assert report["distinct_puzzles"] == 42 and report["distinct_inputs"] == 210
    assert report["stencil_count"] >= 80 and report["output_shape_count"] >= 35
    assert len(report["tile_shapes"]) >= 7 and report["lattice_shape_count"] >= 12
    assert all(len(x) >= 8 for x in report["role_colors"])
    assert report["partial_tile_cases"] == 42 and report["seam_cases"] >= 84
    assert report["fully_ink_erased_tiles"] >= 210
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(140277108, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        _expect_failure(lambda a=a, b=b: generate_puzzle(0, a, b))
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    assert _digest(generate_puzzle(0)) != _digest(generate_puzzle(1))
    # Label damage, duplicate inputs, damaged visible stencil and erased orbits.
    puzzle = generate_puzzle(0, 2, 1)
    out = puzzle["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    _expect_failure(lambda: validate_generated_puzzle(puzzle))
    puzzle = generate_puzzle(1, 2, 1)
    puzzle["test"][0] = {k: v.copy() for k, v in puzzle["train"][0].items()}
    _expect_failure(lambda: validate_generated_puzzle(puzzle))
    puzzle = generate_puzzle(2, 2, 1)
    a = puzzle["test"][0]["input"].copy()
    rec = _decode(a)
    u, v = np.argwhere(rec["stencil"])[0]
    # Change an intact tile's occupied cell to the carrier: periodic occupancy
    # now conflicts with the same local coordinate in other visible tiles.
    p, q = rec["tile_shape"]
    pos = next((1 + i * (p + 1) + int(u), 1 + j * (q + 1) + int(v))
               for i in range(rec["lattice_shape"][0]) for j in range(rec["lattice_shape"][1])
               if a[1 + i * (p + 1) + u, 1 + j * (q + 1) + v] != rec["eraser"])
    a[pos] = rec["carrier"]
    _expect_failure(lambda: solve(a))
    # Fixed family semantics accept symmetric stencils too, even though sampling
    # deliberately uses asymmetric witnesses to reject pixel reflection.
    inks = np.array([[2, 3, 3, 2], [3, 2, 2, 3], [3, 2, 2, 3], [2, 3, 3, 2]], dtype=np.int8)
    stencil = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    full = _paint(inks, stencil, 6, 0)
    a = full.copy()
    a[1:8, 1:8] = 8
    assert np.array_equal(solve(a), full[1:8, 1:8])
    # The central four tiles form one entire erased coarse orbit. Lattice and
    # stencil remain observable, but that orbit's ink cannot be determined.
    erased = full.copy()
    erased[5:12, 5:12] = 8
    _expect_failure(lambda: solve(erased))
    # Recoloring and transposition cannot change the universal rule.
    for pair in generate_puzzle(3)["train"]:
        a, b = pair["input"], pair["output"]
        assert np.array_equal(solve(a.T), b.T)
        perm = np.array([7, 2, 9, 0, 6, 1, 8, 4, 3, 5], dtype=np.int8)
        assert np.array_equal(solve(perm[a]), perm[b])
    return report


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
