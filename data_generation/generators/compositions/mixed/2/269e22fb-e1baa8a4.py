"""Register an exact crop in a visible 20x20 atlas, then collapse full-line runs.

Only standard-library modules and NumPy are used. Nothing is read from disk;
solve receives all reference information in its input. No persistent caches.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _transform(a, index):
    # The eight square isometries: four rotations, then four reflected rotations.
    return np.rot90(np.fliplr(a) if index >= 4 else a, index % 4).copy()


def _binary(a):
    colors = np.unique(a)
    if len(colors) != 2:
        raise ValueError("each panel must contain exactly two colors")
    return (a == colors[1]).astype(np.int8), tuple(int(x) for x in colors)


def _paint(a, colors, swap=0):
    return np.asarray(colors, dtype=np.int8)[a ^ swap]


def _starts(a, axis):
    lines = a if axis == 0 else a.T
    return np.r_[0, 1 + np.flatnonzero(np.any(lines[1:] != lines[:-1], axis=1))]


def _compress(a):
    # Both masks are computed on the full image, not on an individual color run.
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _runs(a, axis):
    return tuple(int(x) for x in np.diff(np.r_[_starts(a, axis), a.shape[axis]]))


def _locations(full, patch):
    h, w = patch.shape
    if h > full.shape[0] or w > full.shape[1]:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    rr, cc = np.nonzero(np.all(windows[:, :, 0, :] == patch[0], axis=-1))
    columns = np.arange(w)
    for r in range(1, h):
        if not len(rr):
            break
        keep = np.all(full[rr[:, None] + r, cc[:, None] + columns] == patch[r], axis=1)
        rr, cc = rr[keep], cc[keep]
    return [(int(r), int(c)) for r, c in zip(rr, cc)]


def _candidates(reference, patch):
    """Unique full answers with their exact offsets; no compressed matching."""
    ref, _ = _binary(reference)
    query, colors = _binary(patch)
    answers = {}
    for orientation in range(8):
        oriented = _transform(ref, orientation)
        for swap in (0, 1):
            offsets = _locations(oriented, query ^ swap)
            if offsets:
                full = _paint(oriented, colors, swap)
                answers.setdefault(_key(full), (full, orientation, swap, offsets))
    return list(answers.values())


def _panels(input_grid):
    """Read two solid, nonzero rectangles separated by zero space."""
    a = _grid(input_grid)
    remaining = set(map(tuple, np.argwhere(a != 0)))
    boxes = []
    while remaining:
        start = remaining.pop()
        stack, component = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    component.append(p)
        r0 = min(p[0] for p in component)
        r1 = max(p[0] for p in component) + 1
        c0 = min(p[1] for p in component)
        c1 = max(p[1] for p in component) + 1
        if len(component) != (r1 - r0) * (c1 - c0):
            raise ValueError("panels must be filled rectangles")
        boxes.append(a[r0:r1, c0:c1].copy())
    if len(boxes) != 2:
        raise ValueError("expected exactly a reference panel and a query panel")
    references = [x for x in boxes if x.shape == (20, 20)]
    if len(references) != 1:
        raise ValueError("expected one 20x20 reference")
    reference = references[0]
    patch = next(x for x in boxes if x is not reference)
    _binary(reference)
    _binary(patch)
    if not (2 <= min(patch.shape) <= 8 and max(patch.shape) <= 20):
        raise ValueError("query dimensions must fit the family crop range")
    if set(np.unique(reference)) & set(np.unique(patch)):
        raise ValueError("reference and query palettes must be disjoint")
    return reference, patch


def _recover(input_grid):
    reference, patch = _panels(input_grid)
    candidates = _candidates(reference, patch)
    if len(candidates) != 1:
        raise ValueError("query must uniquely identify the complete expanded image")
    return reference, patch, candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer orientation and color roles from the crop, complete, then compress."""
    _, _, (full, _, _, _) = _recover(input_grid)
    return _compress(full)


def _observable(a):
    return len(_starts(a, 0)) == a.shape[0] and len(_starts(a, 1)) == a.shape[1]


def _recurrent(a, axis):
    lines = a if axis == 0 else a.T
    return any(np.array_equal(lines[i], lines[j])
               for i in range(len(lines)) for j in range(i + 2, len(lines)))


def _partial_join(a, axis):
    lines = a if axis == 0 else a.T
    same = lines[1:] == lines[:-1]
    return bool(np.any(np.any(same, axis=1) & ~np.all(same, axis=1)))


def _local_alias(full, r, c, h, w):
    """A crop erases a distinction between neighboring complete line vectors."""
    patch = full[r:r + h, c:c + w]
    for axis, start in ((0, r), (1, c)):
        lines = patch if axis == 0 else patch.T
        whole = full if axis == 0 else full.T
        for j in range(1, len(lines)):
            if (np.array_equal(lines[j], lines[j - 1])
                    and not np.array_equal(whole[start + j], whole[start + j - 1])):
                return True
    return False


def _premature_compression_fails(reference, patch, expected):
    answers = _candidates(_compress(reference), _compress(patch))
    return len(answers) != 1 or not np.array_equal(answers[0][0], expected)


def _sample_runs(rng, n):
    # Positive compositions of exactly twenty: no latent boundary is hidden.
    cuts = np.sort(rng.choice(np.arange(1, 20), n - 1, replace=False))
    values = np.diff(np.r_[0, cuts, 20])
    mode = int(rng.integers(4))
    if mode == 1:
        values.sort()
    elif mode == 2:
        values = np.sort(values)[::-1]
    if len(set(values.tolist())) == 1:
        values = values.copy()
        values[0] -= 1
        values[-1] += 1
    return tuple(int(x) for x in values)


def _latent(rng, h, w):
    """Low-resolution bars, walks, rectangles, or perturbed woven patterns."""
    mode = int(rng.integers(4))
    a = np.zeros((h, w), dtype=np.int8)
    if mode == 0:
        # Overlapping short horizontal/vertical bars, including bends and gaps.
        for _ in range(int(rng.integers(h, 2 * h + w))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            if rng.integers(2):
                a[r, min(c, w - 2):min(w, c + int(rng.integers(1, w)))] = 1
            else:
                a[min(r, h - 2):min(h, r + int(rng.integers(1, h))), c] = 1
    elif mode == 1:
        # Alternating regions with a few disruptions, preserving local joins.
        rr = rng.integers(0, 2, h)
        cc = rng.integers(0, 2, w)
        a = (rr[:, None] ^ cc[None, :]).astype(np.int8)
        for _ in range(int(rng.integers(3, h + w))):
            a[int(rng.integers(h)), int(rng.integers(w))] ^= 1
    elif mode == 2:
        # A stepped walk plus smaller branches.
        c = int(rng.integers(w))
        for r in range(h):
            nxt = int(np.clip(c + rng.integers(-2, 3), 0, w - 1))
            a[r, min(c, nxt):max(c, nxt) + 1] = 1
            c = nxt
        for _ in range(h):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            a[r, c:min(w, c + int(rng.integers(1, 4)))] = 1
    else:
        # Painted rectangular patches; overlaps produce broad partial joins.
        for _ in range(2 * (h + w)):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            a[r:min(h, r + int(rng.integers(1, 4))),
              c:min(w, c + int(rng.integers(1, 4)))] = int(rng.integers(2))
    # Deliberately retain separated recurrence on both axes. Placement varies.
    r0 = int(rng.integers(h - 2))
    r1 = int(rng.integers(r0 + 2, h))
    c0 = int(rng.integers(w - 2))
    c1 = int(rng.integers(c0 + 2, w))
    a[r1] = a[r0]
    a[:, c1] = a[:, c0]
    return a


def _reference(rng):
    for _ in range(200):
        h, w = (int(x) for x in rng.integers(5, 10, 2))
        latent = _latent(rng, h, w)
        if not _observable(latent) or not all(_partial_join(latent, x) for x in (0, 1)):
            continue
        orbit = {_key(_transform(latent, i) ^ s) for i in range(8) for s in (0, 1)}
        if len(orbit) != 16:
            continue
        rows, columns = _sample_runs(rng, h), _sample_runs(rng, w)
        expanded = np.repeat(np.repeat(latent, rows, axis=0), columns, axis=1)
        return expanded, latent
    raise RuntimeError("bounded structured-reference sampling exhausted")


def _pack(rng, reference, patch, layout):
    # Query can lie anywhere along the atlas side; rotate/reflect the entire board.
    h, w = patch.shape
    gap = int(rng.integers(1, 3))
    a = np.zeros((20, 20 + gap + w), dtype=np.int8)
    a[:, :20] = reference
    r = int(rng.integers(21 - h))
    a[r:r + h, 20 + gap:] = patch
    return _transform(a, layout)


def _make_pair(rng, reference, orientation, swap, test, witness, seen, max_area=None):
    binary = _transform(reference, orientation)
    # A fresh palette is a bijection; no literal ink color has semantic priority.
    inks = rng.choice(np.arange(1, 10), 4, replace=False)
    ref_colors = tuple(sorted(int(x) for x in inks[:2]))
    query_colors = tuple(sorted(int(x) for x in inks[2:]))
    atlas = _paint(reference, ref_colors)
    target = _paint(binary, query_colors, swap)
    latent = _compress(target)
    for _ in range(500):
        if test:
            h, w = int(rng.integers(10, 21)), int(rng.integers(3, 6))
        else:
            h, w = int(rng.integers(14, 21)), int(rng.integers(6, 9))
        if max_area is not None and h * w >= max_area:
            continue
        r, c = int(rng.integers(21 - h)), int(rng.integers(21 - w))
        patch = target[r:r + h, c:c + w].copy()
        if len(np.unique(patch)) != 2:
            continue
        # Every pair requires completion even after the crop is compressed.
        if _compress(patch).shape == latent.shape:
            continue
        if witness and not _local_alias(target, r, c, h, w):
            continue
        candidates = _candidates(atlas, patch)
        if len(candidates) != 1 or not np.array_equal(candidates[0][0], target):
            continue
        if witness and not _premature_compression_fails(atlas, patch, latent):
            continue
        layout = int(rng.integers(8))
        inp = _pack(rng, atlas, patch, layout)
        out = _transform(latent, layout)
        if _key(inp) in seen:
            continue
        seen.add(_key(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("bounded distinctive-crop sampling exhausted")


def _count(value, minimum, maximum, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= int(value) <= maximum):
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    quarter = int(rng.choice([1, 3]))
    reflection = int(rng.integers(4, 8))
    pool = list(range(8)) if nt >= 3 else list(range(4))
    train_orientations = ([0, quarter, reflection]
                          + [int(rng.choice(pool)) for _ in range(nt)])[:nt]
    # The two-example case demonstrates a rotation generator before holding out
    # a composite rotation. Reflection is used only after a reflection demo.
    unseen = [i for i in pool if i not in train_orientations]
    test_orientations = [int(rng.choice(unseen or pool)) for _ in range(nq)]
    # Crops can become ambiguous in a particular structured atlas. Retry the
    # whole atlas, preserving the one-reference-per-puzzle contract.
    for _ in range(24):
        puzzle, seen = {"train": [], "test": []}, set()
        try:
            reference, _ = _reference(rng)
            for i, orientation in enumerate(train_orientations):
                puzzle["train"].append(_make_pair(
                    rng, reference, orientation, i % 2, False, i == 1, seen))
            train_area = min(_panels(p["input"])[1].size for p in puzzle["train"])
            for i, orientation in enumerate(test_orientations):
                puzzle["test"].append(_make_pair(
                    rng, reference, orientation, int(rng.integers(2)), True, False, seen,
                    max_area=train_area))
        except RuntimeError:
            continue
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("bounded puzzle construction exhausted")


def _canonical(a):
    b, _ = _binary(a)
    return min(_key(_transform(b, i) ^ s) for i in range(8) for s in (0, 1))


def validate_generated_puzzle(puzzle):
    """Inspect returned pixels, never generator metadata or seed state."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, references, training = set(), set(), []
    train_orientations, test_orientations, train_swaps = set(), set(), set()
    witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            reference, patch, (full, orientation, swap, offsets) = _recover(inp)
            references.add(_canonical(reference))
            expected = _compress(full)
            _require(np.array_equal(out, expected), "incorrect compression of recovered image")
            _require(full.shape == (20, 20), "completion size")
            _require(all(5 <= n <= 9 for n in out.shape), "latent dimensions")
            _require(_observable(out), "adjacent output lines must differ")
            _require(all(_recurrent(out, axis) for axis in (0, 1)), "separated recurrence lost")
            _require(all(_partial_join(out, axis) for axis in (0, 1)), "partial joins missing")
            rows, columns = _runs(full, 0), _runs(full, 1)
            _require(sum(rows) == sum(columns) == 20, "full expansion extent")
            _require(len(set(rows)) > 1 and len(set(columns)) > 1, "nonuniform bands required")
            restored = np.repeat(np.repeat(out, rows, axis=0), columns, axis=1)
            _require(np.array_equal(restored, full), "expansion is not lossless")
            _require(np.array_equal(np.unique(out), np.unique(patch)), "query colors changed")
            _require(_compress(patch).shape != out.shape, "completion contribution missing")
            # Full-image compression is substantial in both axes for every pair.
            _require(out.shape[0] < 20 and out.shape[1] < 20, "compression contribution missing")
            h, w = patch.shape
            alias = any(_local_alias(full, r, c, h, w) for r, c in offsets)
            if split == "train":
                training.append((patch.size, min(patch.shape)))
                train_orientations.add(orientation)
                train_swaps.add(swap)
                witness |= alias and _premature_compression_fails(reference, patch, out)
            else:
                test_orientations.add(orientation)
                _require(patch.size < min(x[0] for x in training), "test must show less area")
                _require(min(patch.shape) < min(x[1] for x in training), "test must be narrower")
    _require(len(references) == 1, "all pairs must share one reference orbit")
    _require(witness, "training needs a full-line-versus-local-equality witness")
    _require(0 in train_orientations and bool(train_orientations & {1, 3}),
             "training must expose identity and a quarter turn")
    _require(train_swaps == {0, 1}, "training must expose both role assignments")
    if not train_orientations & {4, 5, 6, 7}:
        _require(test_orientations <= {0, 1, 2, 3}, "test reflection lacks demonstration")


def self_test():
    """42 seeds streamed in memory; no logs, payload files, or execution traces."""
    references, shapes, crops, palettes, bands, orientations = (set() for _ in range(6))
    fingerprints = set()
    local_witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                for a in (inp, out):
                    digest.update(bytes(a.shape))
                    digest.update(a.tobytes())
                reference, patch, (full, orientation, swap, offsets) = _recover(inp)
                references.add(_canonical(reference))
                shapes.add(out.shape)
                crops.add(patch.shape)
                palettes.add(tuple(int(x) for x in np.unique(patch)))
                bands.add((_runs(full, 0), _runs(full, 1)))
                orientations.add(orientation)
                _require(np.array_equal(solve(inp.copy()), out), "solve mismatch")
                # A literal global-dedup shortcut incorrectly deletes separated
                # logical lines. Verify its loss, not just a sampled mode name.
                unique_rows = len({_key(row[None, :]) for row in out})
                unique_cols = len({_key(col[None, :]) for col in out.T})
                _require(unique_rows < out.shape[0] and unique_cols < out.shape[1], "global dedup witness")
                local_witnesses += any(_local_alias(full, r, c, *patch.shape) for r, c in offsets)
        fingerprints.add(digest.digest())
    _require(len(references) == len(fingerprints) == 42, "reference/puzzle diversity")
    _require(len(shapes) >= 15 and len(crops) >= 35, "structural shape diversity")
    _require(len(palettes) >= 25 and len(bands) >= 60, "palette/run diversity")
    _require(orientations == set(range(8)), "all dihedral cases")
    _require(local_witnesses >= 42, "local equality witnesses")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(73001, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(9, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(73002, 7, 3), generate_puzzle(73002, 7, 3)
    _require(all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                 for x, y in zip(a[s], b[s]) for k in ("input", "output")), "determinism")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    colors = np.unique(bad["test"][0]["output"])
    cell = bad["test"][0]["output"][0, 0]
    bad["test"][0]["output"][0, 0] = colors[0] if cell == colors[1] else colors[1]
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Ambiguous monochrome/checkered patches are rejected, never tie-broken.
    checker = np.fromfunction(lambda r, c: (r + c) % 2, (20, 20), dtype=int).astype(np.int8)
    checker[18, 17] ^= 1
    _require(len(_candidates(checker + 1, checker[:4, :4] + 3)) > 1, "ambiguity regression")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "reference_orbits": len(references), "output_shapes": len(shapes),
            "crop_shapes": len(crops), "query_palettes": len(palettes),
            "band_schedules": len(bands), "orientations": sorted(orientations),
            "local_alias_pairs": int(local_witnesses),
            "legal_count_combinations": 30, "independent_solver": False}


if __name__ == "__main__":
    self_test()
    print("self_test passed")
