"""Recognize a framed crop, complete its template, then stably deduplicate columns.

Only numpy and the standard library are used. No files, learned templates,
generation metadata, or previous calls are consulted by solve().
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


BACKGROUND = 0
REFERENCE_FRAME = 8
QUERY_FRAME = 9
DATA_COLORS = np.arange(1, 8, dtype=np.int8)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= v <= 30 for v in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _transform(a, index):
    # 0..3: counterclockwise turns; 4..7: left-right reflection, then turns.
    return np.rot90(a if index < 4 else np.fliplr(a), index % 4).copy()


def _binary(a):
    colors = np.unique(a)
    if len(colors) != 2:
        raise ValueError("each panel must contain exactly two colors")
    return (a == colors[1]).astype(np.int8)


def _panel(canvas, color):
    rr, cc = np.nonzero(canvas == color)
    if not len(rr):
        raise ValueError("missing frame")
    top, bottom, left, right = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    box = canvas[top:bottom + 1, left:right + 1]
    if (min(box.shape) < 4 or not np.all(box[0] == color)
            or not np.all(box[-1] == color) or not np.all(box[:, 0] == color)
            or not np.all(box[:, -1] == color)):
        raise ValueError("frame must be a complete rectangle")
    inside = box[1:-1, 1:-1]
    if not np.all((inside >= 1) & (inside <= 7)):
        raise ValueError("panel interiors use colors 1..7")
    _binary(inside)
    return inside.copy(), (top, bottom + 1, left, right + 1)


def _parse(input_grid):
    canvas = _grid(input_grid)
    reference, a = _panel(canvas, REFERENCE_FRAME)
    query, b = _panel(canvas, QUERY_FRAME)
    used = np.zeros(canvas.shape, dtype=bool)
    for top, bottom, left, right in (a, b):
        if np.any(used[top:bottom, left:right]):
            raise ValueError("panels overlap")
        used[top:bottom, left:right] = True
    if np.any(canvas[~used] != BACKGROUND):
        raise ValueError("outside panels must be black")
    return reference, query


def _has_window(full, patch):
    if any(x > y for x, y in zip(patch.shape, full.shape)):
        return False
    windows = np.lib.stride_tricks.sliding_window_view(full, patch.shape)
    return bool(np.any(np.all(windows == patch, axis=(-1, -2))))


def _matches(reference, patch):
    """Return distinct complete binary images and their orientation/role labels.

    Repeated offsets yielding the same complete image are harmless. Different
    completions are rejected even when their compressed outputs would coincide.
    """
    found = {}
    for orientation in range(8):
        full = _transform(reference, orientation)
        for swap in (0, 1):
            candidate = full if swap == 0 else 1 - full
            if _has_window(candidate, patch):
                found[_key(candidate)] = (candidate, orientation, swap)
    return list(found.values())


def _recover(reference, query):
    colors = np.unique(query)
    matches = _matches(_binary(reference), _binary(query))
    if len(matches) != 1:
        raise ValueError("crop must identify exactly one complete image")
    mask, orientation, swap = matches[0]
    if query.size >= mask.size:
        raise ValueError("query must be a proper crop")
    return colors[mask].astype(np.int8), orientation, swap


def _unique_columns(a):
    seen, indices = set(), []
    for j in range(a.shape[1]):
        key = a[:, j].tobytes()
        if key not in seen:
            seen.add(key)
            indices.append(j)
    return a[:, indices].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use visible reference/crop evidence, then scan recovered columns left to right."""
    reference, query = _parse(input_grid)
    completed, _, _ = _recover(reference, query)
    return _unique_columns(completed)


def _sequence(rng, n):
    order = [int(v) for v in rng.permutation(n)]
    a, b = order[:2]
    seq = [a, b, a] + order[2:]
    # Extra copies involve only the first two types: the final novel row type
    # remains a singleton and can be hidden in the full-vector witness crop.
    extras = int(rng.integers(0, 13 - len(seq) - 2))
    for _ in range(extras):
        at = int(rng.integers(0, len(seq) + 1))
        seq.insert(at, int(rng.choice((a, b))))
    seq += [b, a]
    return seq, order


def _draw_core(rng, h, w, near_columns, hidden_row):
    a = np.zeros((h, w), dtype=np.int8)
    for r in range(h):
        x, y = sorted(int(v) for v in rng.choice(w + 1, 2, replace=False))
        a[r, x:y] = 1
    for _ in range(int(rng.integers(1, 5))):
        c = int(rng.integers(w))
        x, y = sorted(int(v) for v in rng.choice(h + 1, 2, replace=False))
        a[x:y, c] ^= 1
    if rng.random() < .5:
        for r in range(h):
            a[r, (r + int(rng.integers(2))) % w] ^= 1
    # These columns agree on every row except the one omitted by a witness.
    first, second = near_columns
    a[:, second] = a[:, first]
    a[hidden_row, second] ^= 1
    return a


def _crop_bank(rng, reference, low, high, limit=8):
    H, W = reference.shape
    result, seen = [], set()
    # Randomized finite search, followed by exhaustive shape/offset enumeration.
    shapes = [(h, w) for h in range(2, H) for w in range(2, W)
              if low <= h * w / reference.size <= high]
    rng.shuffle(shapes)
    for h, w in shapes:
        accepted_for_shape = 0
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        rng.shuffle(positions)
        for r, c in positions:
            patch = reference[r:r + h, c:c + w]
            key = _key(patch)
            if key in seen or len(np.unique(patch)) != 2:
                continue
            seen.add(key)
            if len(_matches(reference, patch)) == 1:
                result.append((r, c, h, w))
                accepted_for_shape += 1
                if len(result) == limit:
                    return result
                if accepted_for_shape == 2:
                    break
    return result


def _prepare_reference(rng, reference, hidden_at):
    # All 16 orientation/role images must differ before searching proper crops.
    orbit = {_key(_transform(reference, i) ^ s) for i in range(8) for s in (0, 1)}
    if len(orbit) != 16:
        return None
    witness = (0, 0, hidden_at, reference.shape[1] - 1)
    patch = reference[:hidden_at, :-1]
    if len(np.unique(patch)) != 2 or len(_matches(reference, patch)) != 1:
        return None
    large = _crop_bank(rng, reference, .55, .9)
    small = _crop_bank(rng, reference, .22, .5)
    if len(large) < 4 or len(small) < 4:
        return None
    return reference, witness, large, small


def _reference(rng):
    for _ in range(96):
        h, w = (int(v) for v in rng.integers(4, 8, size=2))
        rows, ro = _sequence(rng, h)
        cols, co = _sequence(rng, w)
        core = _draw_core(rng, h, w, co[:2], ro[-1])
        if (len({row.tobytes() for row in core}) != h
                or len({col.tobytes() for col in core.T}) != w):
            continue
        reference = core[np.ix_(rows, cols)]
        prepared = _prepare_reference(rng, reference, rows.index(ro[-1]))
        if prepared is not None:
            return prepared
    # Explicit finite fallback with the same structure and no solver-side bank.
    core = np.array([[int(c) for c in row] for row in
                     ("001011", "000101", "001110", "001001", "010110")], dtype=np.int8)
    rows, cols = [0, 1, 0, 2, 3, 4, 1, 0], [0, 1, 0, 2, 3, 4, 5, 1, 0]
    prepared = _prepare_reference(rng, core[np.ix_(rows, cols)], 5)
    if prepared is None:
        raise RuntimeError("fallback reference failed its visible-evidence checks")
    return prepared


def _frame(a, color):
    framed = np.full((a.shape[0] + 2, a.shape[1] + 2), color, dtype=np.int8)
    framed[1:-1, 1:-1] = a
    return framed


def _pack(rng, reference, query):
    panels = [_frame(reference, REFERENCE_FRAME), _frame(query, QUERY_FRAME)]
    if rng.random() < .5:
        panels.reverse()
    horizontal = bool(rng.integers(2))
    if not horizontal:
        panels = [a.T for a in panels]
    gap = int(rng.integers(1, 3))
    height = max(a.shape[0] for a in panels)
    width = sum(a.shape[1] for a in panels) + gap
    # At most 30: interiors have sides <=12, frames add 4, gap adds <=2.
    extra_h = int(rng.integers(0, min(2, 30 - height) + 1))
    extra_w = int(rng.integers(0, min(2, 30 - width) + 1))
    canvas = np.zeros((height + extra_h, width + extra_w), dtype=np.int8)
    c = int(rng.integers(extra_w + 1))
    for panel in panels:
        r = int(rng.integers(canvas.shape[0] - panel.shape[0] + 1))
        canvas[r:r + panel.shape[0], c:c + panel.shape[1]] = panel
        c += panel.shape[1] + gap
    # Transposing the packed canvas back preserves both panel interiors.
    return canvas if horizontal else canvas.T.copy()


def _palette(rng):
    return rng.choice(DATA_COLORS, size=2, replace=False)


def _make_pair(rng, reference, crop, orientation, query_colors, reference_colors, seen):
    r, c, h, w = crop
    completed_mask = _transform(reference, orientation)
    patch_mask = _transform(reference[r:r + h, c:c + w], orientation)
    query = query_colors[patch_mask]
    displayed = reference_colors[reference]
    # Label construction does not call solve or the production compression helper.
    columns = [tuple(int(v) for v in col) for col in query_colors[completed_mask].T]
    output = np.array(list(dict.fromkeys(columns)), dtype=np.int8).T.copy()
    for _ in range(32):
        canvas = _pack(rng, displayed, query)
        if _key(canvas) not in seen:
            seen.add(_key(canvas))
            return {"input": canvas, "output": output}
    raise RuntimeError("could not pack a distinct input within 32 attempts")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    reference, witness, large, small = _reference(rng)
    quarter = int(rng.choice((1, 3)))
    anchors = [0, quarter]
    if nt >= 3:
        anchors.append(4)
    allowed = list(range(8)) if nt >= 3 else list(range(4))
    held_out = int(rng.choice([i for i in allowed if i not in anchors]))
    train_orientations = anchors + [int(rng.choice([i for i in allowed if i != held_out]))
                                    for _ in range(nt - len(anchors))]
    test_orientations = [held_out] + [int(rng.choice(allowed)) for _ in range(nq - 1)]
    # A repeated two-color palette with exchanged roles explicitly teaches that
    # the query pattern, rather than color magnitude, determines the role map.
    first_colors, reference_colors = _palette(rng), _palette(rng)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, orientations in (("train", train_orientations), ("test", test_orientations)):
        for index, orientation in enumerate(orientations):
            if split == "train" and index == 0:
                crop, palette = witness, first_colors
            elif split == "train" and index == 1:
                crop, palette = large[int(rng.integers(len(large)))], first_colors[::-1]
            else:
                bank = small if split == "test" else large
                crop, palette = bank[int(rng.integers(len(bank)))], _palette(rng)
            shown_colors = reference_colors if index < 2 and split == "train" else _palette(rng)
            puzzle[split].append(_make_pair(rng, reference, crop, orientation,
                                           palette, shown_colors, seen))
    return puzzle


def _column_keys(a):
    return [tuple(int(v) for v in col) for col in a.T]


def _ablation_flags(reference, query, completed, output, orientation):
    columns = _column_keys(completed)
    adjacent = [columns[0]] + [columns[j] for j in range(1, len(columns))
                               if columns[j] != columns[j - 1]]
    last = [col for j, col in enumerate(columns) if col not in columns[j + 1:]]
    n = output.shape[1]
    # Try BOTH possible role bijections when skipping crop recognition entirely.
    ref_mask, colors = _binary(reference), np.unique(query)
    no_recognition = all(not np.array_equal(_unique_columns(colors[ref_mask ^ swap]), output)
                         for swap in (0, 1))
    return dict(no_completion=not np.array_equal(_unique_columns(query), output),
                no_dedup=not np.array_equal(completed, output),
                adjacent_only=len(adjacent) != n,
                prefix=not np.array_equal(completed[:, :n], output),
                last_occurrence=last != _column_keys(output),
                sorted_vectors=sorted(set(columns)) != _column_keys(output),
                both_axes=len({row.tobytes() for row in output}) != output.shape[0],
                compress_before_geometry=all(
                    not np.array_equal(_transform(_unique_columns(colors[ref_mask ^ swap]), orientation), output)
                    for swap in (0, 1)),
                no_recognition=no_recognition)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, reference_key, train_orientations, train_flags = set(), None, set(), []
    records = {"train": [], "test": []}
    collision_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate inputs")
            seen.add(_key(inp))
            reference, query = _parse(inp)
            mask = _binary(reference)
            canonical = min(_key(mask), _key(1 - mask))
            if reference_key is None:
                reference_key = canonical
            _require(canonical == reference_key, "reference content changes within the puzzle")
            _require(7 <= min(reference.shape) <= max(reference.shape) <= 12, "reference side range")
            _require(query.shape[0] < max(reference.shape) and query.shape[1] < max(reference.shape),
                     "query must omit cells on both axes")
            completed, orientation, swap = _recover(reference, query)
            _require(all(q < t for q, t in zip(query.shape, completed.shape)), "crop must omit both axes")
            # Independent quadratic visible-vector comparison validates labels.
            columns = _column_keys(completed)
            keep = [j for j, col in enumerate(columns) if col not in columns[:j]]
            expected = completed[:, keep]
            _require(np.array_equal(out, expected), "wrong first-occurrence output")
            _require(np.array_equal(solve(inp), out), "input-only solver disagrees")
            _require(set(np.unique(out)) == set(np.unique(query)), "query colors not preserved")
            _require(len(set(_column_keys(out))) == out.shape[1], "duplicate output columns")
            flags = _ablation_flags(reference, query, completed, out, orientation)
            _require(all(flags[k] for k in ("no_completion", "no_dedup", "adjacent_only", "both_axes")),
                     "an inherited mechanism or whole-axis behavior is vacuous")
            if split == "train":
                train_orientations.add(orientation)
                train_flags.append(flags)
                if orientation == 0:
                    # Inspect exact crop offsets; matching on only query rows
                    # conflates two globally different columns in this witness.
                    H, W = completed.shape
                    h, w = query.shape
                    for r in range(H - h + 1):
                        for c in range(W - w + 1):
                            if np.array_equal(completed[r:r + h, c:c + w], query):
                                part, full = _column_keys(query), _column_keys(completed[:, c:c + w])
                                collision_witness |= any(part[i] == part[j] and full[i] != full[j]
                                                         for i in range(w) for j in range(i))
            records[split].append((orientation, swap, query.size / completed.size))
    _require({0} <= train_orientations and bool(train_orientations & {1, 3}), "missing geometry anchors")
    _require(collision_witness, "training lacks a partial-column collision witness")
    for flag in ("prefix", "last_occurrence", "no_recognition", "compress_before_geometry"):
        _require(any(f[flag] for f in train_flags), f"training does not defeat {flag}")
    _require({r[1] for r in records["train"]} == {0, 1}, "both color-role maps must be demonstrated")
    _require(records["test"][0][0] not in train_orientations, "first test must hold out an orientation")
    _require(records["test"][0][2] < records["train"][1][2], "test crop shift missing")
    _require(all(.22 <= r[2] <= .5 for r in records["test"]), "test crop fraction range")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, all legal counts, ablations, determinism, corruption.

    No files or execution traces are written; retained metrics contain only
    small signatures/sets, not a reconstructible corpus of generated pairs.
    """
    signatures, references, shapes, outputs, crops, palettes, orientations = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        reference = _binary(_parse(puzzle["train"][0]["input"])[0])
        canonical = min(_key(_transform(reference, i) ^ s) for i in range(8) for s in (0, 1))
        references.add(hashlib.sha256(canonical[1]).hexdigest())
        for pair in puzzle["train"] + puzzle["test"]:
            ref, query = _parse(pair["input"])
            _, orientation, _ = _recover(ref, query)
            orientations.add(orientation)
            shapes.add(pair["input"].shape)
            outputs.add(pair["output"].shape)
            crops.add(query.shape)
            palettes.add(tuple(int(c) for c in np.unique(query)))
        if seed in (0, 17, 41):
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
    _require(len(signatures) == 42 and len(references) >= 40, "reference diversity")
    _require(len(shapes) >= 35 and len(outputs) >= 15 and len(crops) >= 25, "geometry diversity")
    _require(len(palettes) >= 18 and orientations == set(range(8)), "palette/orientation diversity")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, nq), "counts ignored")
            validate_generated_puzzle(puzzle)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Force the documented finite fallback's evidence check too.
    core = np.array([[int(c) for c in row] for row in
                     ("001011", "000101", "001110", "001001", "010110")], dtype=np.int8)
    ref = core[np.ix_([0, 1, 0, 2, 3, 4, 1, 0], [0, 1, 0, 2, 3, 4, 5, 1, 0])]
    _require(_prepare_reference(np.random.default_rng(0), ref, 5) is not None, "fallback invalid")
    try:
        _recover(ref + 1, np.array([[1, 2], [1, 2]], dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous crop accepted")
    puzzle = generate_puzzle(0)
    puzzle["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal count combinations)")
