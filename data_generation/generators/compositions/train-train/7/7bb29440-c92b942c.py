"""Periodic sieve: compare residual grains after broadcasting corner seed motifs.

Only NumPy and the Python standard library are used. All semantic information
used by solve is present in the input; generation plans are never consulted.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 278370779, 666158075, 105245473, 405864335, 896179669,
              154294026, 828557258, 601152047, 261814242, 710601678,
              674635784, 805597805, 146468240, 542497653, 223357219,
              580747400, 717376289, 688884424, 209755816, 880024504,
              685288486, 49296926, 721789835, 411915020, 916790885,
              178741918, 636159892, 818810395, 912493390, 618344358,
              449558100, 999191877, 469131089, 842508803, 168843601,
              108321053, 799768793, 948793962, 659858344, 50012205,
              125710966)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0) or np.any(array > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return array


def _key(grid):
    return grid.shape, grid.tobytes()


def _boxes(grid, background):
    """Four-connected components of complete non-background rectangles."""
    occupied = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    boxes = []
    height, width = grid.shape
    for row, col in np.argwhere(occupied):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        stack, cells = [(row, col)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= y < height and 0 <= x < width
                        and occupied[y, x] and not seen[y, x]):
                    seen[y, x] = True
                    stack.append((y, x))
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells) + 1
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells) + 1
        if len(cells) != (r1 - r0) * (c1 - c0):
            raise ValueError("candidate is not a solid rectangle")
        boxes.append((r0, r1, c0, c1))
    return boxes


def _sieve(unit):
    """Return copied seeds, row sweep, diagonal-only sweep, and seam cells."""
    seeds = np.tile(unit, (3, 3))
    rows = np.broadcast_to(np.any(seeds, axis=1)[:, None], seeds.shape).copy()
    far_rows = np.zeros(seeds.shape, dtype=bool)
    for row in np.flatnonzero(np.any(seeds, axis=1)):
        columns = np.flatnonzero(seeds[row])
        distances = np.abs(np.arange(seeds.shape[1])[:, None] - columns[None, :])
        far_rows[row] = np.min(distances, axis=1) >= 2
    diagonals = np.zeros(seeds.shape, dtype=bool)
    seams = np.zeros(seeds.shape, dtype=bool)
    opposite = np.zeros(seeds.shape, dtype=bool)
    h, w = unit.shape
    height, width = seeds.shape
    for r, c in np.argwhere(seeds):
        r, c = int(r), int(c)
        for d in (-1, 1):
            y, x = r + d, c + d
            if 0 <= y < height and 0 <= x < width:
                diagonals[y, x] = True
                if y // h != r // h or x // w != c // w:
                    seams[y, x] = True
            y, x = r + d, c - d
            if 0 <= y < height and 0 <= x < width:
                opposite[y, x] = True
    diagonals &= ~rows
    seams &= diagonals
    footprint = rows | diagonals  # rows already include copied seed positions
    opposite &= ~footprint
    return {"seeds": seeds, "rows": rows, "far_rows": far_rows, "diagonals": diagonals,
            "seams": seams, "opposite": opposite, "footprint": footprint}


def _read(input_grid):
    grid = _grid(input_grid)
    colors, frequencies = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(frequencies))])
    if frequencies.max() * 2 <= grid.size or len(colors) != 4:
        raise ValueError("expected four colors and a strict background majority")
    boxes = _boxes(grid, background)
    if not 2 <= len(boxes) <= 5:
        raise ValueError("expected two through five swatches")
    patches, carriers = [], []
    for r0, r1, c0, c1 in boxes:
        patch = grid[r0:r1, c0:c1]
        if any(n % 3 or not 6 <= n <= 15 for n in patch.shape):
            raise ValueError("swatch sides must be three times a period of 2..5")
        hues, counts = np.unique(patch, return_counts=True)
        if counts.max() * 2 <= patch.size:
            raise ValueError("each swatch needs a strict carrier majority")
        carriers.append(int(hues[int(np.argmax(counts))]))
        patches.append(patch)
    if len(set(carriers)) != 1:
        raise ValueError("swatches must share a carrier")
    carrier = carriers[0]
    auxiliary = set(int(c) for c in colors) - {background, carrier}
    possible = []
    for hue in auxiliary:
        confined = True
        for patch in patches:
            h, w = patch.shape[0] // 3, patch.shape[1] // 3
            positions = np.argwhere(patch == hue)
            if (not 1 <= len(positions) <= 4
                    or np.any(positions[:, 0] >= h) or np.any(positions[:, 1] >= w)):
                confined = False
                break
        if confined:
            possible.append(hue)
    if len(possible) != 1:
        raise ValueError("corner confinement must identify one seed hue")
    seed_color = possible[0]
    grain_color = (auxiliary - {seed_color}).pop()
    records = []
    for box, patch in zip(boxes, patches):
        h, w = patch.shape[0] // 3, patch.shape[1] // 3
        unit = patch[:h, :w] == seed_color
        masks = _sieve(unit)
        grains = patch == grain_color
        beyond = grains.copy()
        beyond[:h, :w] = False
        if not beyond.any():
            raise ValueError("grain hue must appear beyond the seed sample")
        residue = grains & ~masks["footprint"]
        records.append({"box": box, "patch": patch, "unit": unit,
                        "grains": grains, "residue": residue,
                        "score": int(residue.sum()), "raw": int(grains.sum()),
                        "area": patch.size, **masks})
    scores = [r["score"] for r in records]
    if scores.count(min(scores)) != 1:
        raise ValueError("residue minimum must be unique")
    return {"background": background, "carrier": carrier, "seed_color": seed_color,
            "grain_color": grain_color, "records": records,
            "selected": int(np.argmin(scores))}


def _finish(record, roles):
    output = np.full(record["patch"].shape, roles["carrier"], dtype=np.int8)
    output[record["residue"]] = roles["grain_color"]
    output[record["seeds"]] = roles["seed_color"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    roles = _read(input_grid)
    return _finish(roles["records"][roles["selected"]], roles)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _sample_mask(rng, mask, count, mandatory=()):
    selected = set((int(r), int(c)) for r, c in mandatory)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in selected]
    if count < len(selected) or count - len(selected) > len(cells):
        return None
    if count > len(selected):
        indices = rng.choice(len(cells), size=count - len(selected), replace=False)
        selected.update(cells[int(i)] for i in np.atleast_1d(indices))
    return selected


def _motif(rng, shape, kind, seed_cap):
    h, w = shape
    unit = np.zeros(shape, dtype=bool)
    if kind == "row":
        row = int(rng.integers(h))
        edge = (0, w - 1)[int(rng.integers(2))]
        other = _pick(rng, [c for c in range(w) if c != edge])
        unit[row, [edge, other]] = True
    elif kind == "multi":
        row, col = int(rng.integers(h - 1)), int(rng.integers(w - 1))
        unit[row, col] = unit[row + 1, col + 1] = True
    else:
        count = (seed_cap if kind == "crowded" else
                 int(rng.integers(1, min(seed_cap, h * w // 4) + 1)))
        slots = rng.choice(h * w, size=count, replace=False)
        unit.flat[slots] = True
    return unit


def _swatch(rng, shape, kind, score, covered, palette, seed_cap, diagnostic):
    carrier, seed_color, grain_color = palette[1:]
    for _ in range(120):
        unit = _motif(rng, shape, kind, seed_cap)
        masks = _sieve(unit)
        h, w = shape
        original = np.zeros(masks["seeds"].shape, dtype=bool)
        original[:h, :w] = unit
        free_covered = masks["footprint"] & ~original
        free_residue = ~masks["footprint"]
        required_covered, required_residue = set(), set()
        if diagnostic:
            # These three changes distinguish a row-only sweep, independent
            # tile decoration, and failing to broadcast the seed itself.
            row_witness = (masks["far_rows"] if masks["far_rows"].any()
                           else masks["rows"] & ~masks["seeds"])
            pools = [row_witness,
                     masks["diagonals"], masks["seams"],
                     masks["seeds"] & ~original]
            if not all(pool.any() for pool in pools) or not masks["opposite"].any():
                continue
            for pool in pools:
                required_covered.add(_pick(rng, [tuple(map(int, p)) for p in np.argwhere(pool)]))
            required_residue.add(_pick(rng, [tuple(map(int, p)) for p in np.argwhere(masks["opposite"])]))
        if covered + score + int(unit.sum()) >= masks["seeds"].size / 2:
            return None
        hit = _sample_mask(rng, free_covered, covered, required_covered)
        miss = _sample_mask(rng, free_residue, score, required_residue)
        if hit is None or miss is None:
            continue
        grains = hit | miss
        if not any(r >= h or c >= w for r, c in grains):
            continue
        patch = np.full(masks["seeds"].shape, carrier, dtype=np.int8)
        for r, c in grains:
            patch[r, c] = grain_color
        patch[original] = seed_color
        return patch
    return None


def _apart(rect, placed):
    r, c, h, w = rect
    return all(r + h < y or y + hh < r or c + w < x or x + ww < c
               for y, x, hh, ww in placed)


def _pack(rng, patches, background):
    area = sum(p.size for p in patches)
    max_h, max_w = max(p.shape[0] for p in patches), max(p.shape[1] for p in patches)
    for _ in range(55):
        height = int(rng.integers(max(14, max_h), 31))
        low_width = max(14, max_w, int(np.ceil(area / (.45 * height))))
        if low_width > 30:
            continue
        width = int(rng.integers(low_width, 31))
        order = sorted(range(len(patches)), key=lambda i: patches[i].size, reverse=True)
        rectangles, placed = [None] * len(patches), []
        for i in order:
            h, w = patches[i].shape
            cells = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
            for j in rng.permutation(len(cells)):
                r, c = cells[int(j)]
                rect = (r, c, h, w)
                if _apart(rect, placed):
                    placed.append(rect)
                    rectangles[i] = rect
                    break
            if rectangles[i] is None:
                break
        if all(rect is not None for rect in rectangles):
            grid = np.full((height, width), background, dtype=np.int8)
            for patch, (r, c, h, w) in zip(patches, rectangles):
                grid[r:r + h, c:c + w] = patch
            return grid
    return None


def _make_pair(rng, palette, profile, shift, training):
    if training:
        count = int(rng.integers(2, 4))
    else:
        count = int(rng.integers(4, 6)) if shift == "count" else int(rng.integers(2, 4))
    max_period = 4 if training or shift == "count" else 5
    shape_pool = [(h, w) for h in range(2, max_period + 1)
                  for w in range(2, max_period + 1)
                  if count < 4 or h * w <= 9]
    for _ in range(180):
        if profile == "inversion":
            target_shape = _pick(rng, [(3, 3), (3, 4), (4, 3), (4, 4)])
            minimum = int(rng.integers(1, 4))
            target_kind, target_covered = "row", int(rng.integers(10, 16))
            rival_pool = [s for s in shape_pool if s[0] * s[1] < np.prod(target_shape)]
        elif profile == "ordinary":
            target_shape = _pick(rng, [(3, 3), (3, 4), (4, 3)])
            minimum = 4
            target_kind, target_covered = "multi", int(rng.integers(4, 7))
            rival_pool = [s for s in shape_pool
                          if s[0] * s[1] * minimum > np.prod(target_shape) * (minimum + 1)]
        else:
            target_shape = _pick(rng, shape_pool)
            if not training and shift == "period":
                target_shape = _pick(rng, [s for s in shape_pool if 5 in s])
            if not training and shift == "motif":
                target_shape = _pick(rng, [s for s in shape_pool if min(s) >= 3 and s[0] * s[1] >= 12])
            minimum = int(rng.integers(0, 5))
            target_kind = "crowded" if not training and shift == "motif" else "random"
            target_covered = int(rng.integers(2, 14))
            rival_pool = shape_pool
        if not rival_pool:
            continue
        seed_cap = 2 if training and shift == "motif" else (3 if target_kind == "crowded" else 4)
        patches = []
        target = _swatch(rng, target_shape, target_kind, minimum, target_covered,
                         palette, seed_cap, profile in ("inversion", "ordinary"))
        if target is None:
            continue
        patches.append(target)
        for j in range(count - 1):
            shape = _pick(rng, rival_pool)
            score = minimum + 1 if j == 0 else minimum + int(rng.integers(1, 5))
            covered = (int(rng.integers(1, 5)) if profile == "inversion" else
                       int(rng.integers(8, 13)) if profile == "ordinary" else
                       int(rng.integers(1, 13)))
            patch = _swatch(rng, shape, "random", score, covered, palette,
                            2 if training and shift == "motif" else 4, False)
            if patch is None:
                break
            patches.append(patch)
        if len(patches) != count:
            continue
        order = rng.permutation(count)
        grid = _pack(rng, [patches[int(i)] for i in order], palette[0])
        if grid is None:
            continue
        info = _read(grid)
        winner = info["records"][info["selected"]]
        others = [r for i, r in enumerate(info["records"]) if i != info["selected"]]
        if profile == "inversion" and not any(winner["raw"] > r["raw"] for r in others):
            continue
        if profile == "ordinary" and not all(winner["raw"] < r["raw"] for r in others):
            continue
        return {"input": grid, "output": _finish(winner, info)}
    raise RuntimeError("bounded swatch construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not 2 <= num_train <= 10
            or not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, with train > test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(c) for c in rng.choice(10, size=4, replace=False))
    shift = _pick(rng, ("count", "period", "motif"))
    demonstrations = ["inversion", "ordinary"]
    rng.shuffle(demonstrations)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            profile = demonstrations[index] if split == "train" and index < 2 else "random"
            for _ in range(30):
                pair = _make_pair(rng, palette, profile, shift, split == "train")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _diagnostics(record):
    grains = record["grains"]
    h, w = record["unit"].shape
    original = np.zeros(grains.shape, dtype=bool)
    original[:h, :w] = record["unit"]
    return (bool(np.any(grains & record["far_rows"])),
            bool(np.any(grains & record["diagonals"])),
            bool(np.any(grains & record["seams"])),
            bool(np.any(grains & record["seeds"] & ~original)),
            bool(np.any(grains & record["opposite"])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    analyses, seen, roles = {"train": [], "test": []}, set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            # Normalize before using bytes, so a list and an array cannot evade uniqueness.
            key = _key(grid.astype(np.int8))
            assert key not in seen
            seen.add(key)
            info = _read(grid)
            roles.add(tuple(info[k] for k in ("background", "carrier", "seed_color", "grain_color")))
            records = info["records"]
            winner = records[info["selected"]]
            expected = _finish(winner, info)
            assert np.array_equal(output, expected)
            assert int(np.count_nonzero(output == info["grain_color"])) == winner["score"]
            assert int(np.count_nonzero(output == info["seed_color"])) == 9 * int(winner["unit"].sum())
            assert not np.any(winner["residue"] & winner["footprint"])
            assert np.all(winner["residue"] <= winner["grains"])
            assert all(r["score"] > winner["score"] for i, r in enumerate(records) if i != info["selected"])
            assert output.shape == winner["patch"].shape
            analyses[split].append(info)
    assert len(roles) == 1
    train = analyses["train"]
    chosen = [a["records"][a["selected"]] for a in train]
    comparisons = [(a["records"][a["selected"]], [r for i, r in enumerate(a["records"]) if i != a["selected"]]) for a in train]
    assert any(any(w["raw"] > r["raw"] for r in others) for w, others in comparisons)
    assert any(all(w["raw"] < r["raw"] for r in others) for w, others in comparisons)
    assert any(all(w["area"] < r["area"] for r in others) for w, others in comparisons)
    assert any(all(w["area"] > r["area"] for r in others) for w, others in comparisons)
    assert any(any(r["score"] * w["area"] < w["score"] * r["area"] for r in others) for w, others in comparisons)
    assert all(any(_diagnostics(w)[i] for w in chosen) for i in range(5))
    assert any(any(w["unit"].sum(axis=1) >= 2) for w in chosen)
    assert any(np.count_nonzero(w["unit"].any(axis=1)) >= 2 for w in chosen)
    # At least one transparent structural shift, inferred from the grids.
    all_train = [r for a in train for r in a["records"]]
    test_chosen = [a["records"][a["selected"]] for a in analyses["test"]]
    count_shift = max(len(a["records"]) for a in analyses["test"]) > max(len(a["records"]) for a in train)
    period_shift = (max(max(r["unit"].shape) for r in test_chosen)
                    > max(max(r["unit"].shape) for r in all_train))
    motif_shift = max(int(r["unit"].sum()) for r in test_chosen) > max(int(r["unit"].sum()) for r in all_train)
    assert count_shift or period_shift or motif_shift


def _fingerprint(puzzle):
    return tuple((split, tuple((_key(np.asarray(p["input"], dtype=np.int8)),
                               _key(np.asarray(p["output"], dtype=np.int8))) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test():
    """Bounded checks on 42 published seeds, all counts, and hand-derived masks."""
    # A 3x3 unit with one corner seed. NW from its copy at (3,3)
    # reaches (2,2), whereas independently sweeping each tile misses it.
    unit = np.zeros((3, 3), dtype=bool)
    unit[0, 0] = True
    masks = _sieve(unit)
    expected = np.zeros((9, 9), dtype=bool)
    expected[[0, 3, 6], :] = True
    for r in (0, 3, 6):
        for c in (0, 3, 6):
            if r and c:
                expected[r - 1, c - 1] = True
            if r + 1 < 9 and c + 1 < 9:
                expected[r + 1, c + 1] = True
    assert np.array_equal(masks["footprint"], expected)
    assert masks["seams"][2, 2] and not masks["footprint"][1, 2]
    assert masks["far_rows"][0, 8] and not masks["far_rows"][0, 1]
    puzzles, shapes, motifs, palettes, counts, scores, outputs = set(), set(), set(), set(), set(), set(), set()
    edge_contact = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        puzzles.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                info = _read(pair["input"])
                palettes.add(tuple(info[k] for k in ("background", "carrier", "seed_color", "grain_color")))
                counts.add(len(info["records"]))
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                for r in info["records"]:
                    motifs.add((r["unit"].shape, tuple(map(tuple, np.argwhere(r["unit"])))))
                    scores.add(r["score"])
                    r0, r1, c0, c1 = r["box"]
                    edge_contact.add(r0 == 0 or c0 == 0 or r1 == pair["input"].shape[0] or c1 == pair["input"].shape[1])
    assert len(puzzles) == 42 and len(palettes) >= 30
    assert len(shapes) >= 30 and len(outputs) >= 10 and len(motifs) >= 100
    assert counts == {2, 3, 4, 5} and {0, 1, 2, 3, 4}.issubset(scores)
    assert edge_contact == {False, True}
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(12345 + nt * 10 + ns, nt, ns)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
    assert _fingerprint(generate_puzzle(987, 5, 3)) == _fingerprint(generate_puzzle(987, 5, 3))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 0.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, deterministic masks and structural variation")
