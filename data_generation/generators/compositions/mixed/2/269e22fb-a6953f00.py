"""Faithful implementation of proposal.json, including its parity degeneracy.

One fixed 20x20 binary reference is a FAMILY CONSTANT, not a stored example
or seed-dependent state. Matching reconstructs its orientation and color roles
before selecting a corner. Every completion is even: always-right is therefore
observationally equivalent to the requested parity selector. See composition.md.
Generation, validation and self_test never write files or execution traces.
"""
from __future__ import annotations

import numpy as np


# The original A reference, with its two literal colors replaced by binary roles.
# The first two generated training INPUTS jointly expose a transformed copy.
REFERENCE_ROWS = (
    "00011000000000000011", "00011000000000111111",
    "00111000000001111111", "00111000000011100011",
    "01111100000111000000", "01100100001110000000",
    "01100111001100000000", "01100101001000000000",
    "01100101111000000000", "01100101101111111110",
    "01100101111000000010", "01100101001111111110",
    "01100111001000001010", "01100100001111110111",
    "01111100001000011010", "00110110001000010111",
    "01100010001000011010", "11000110001000010111",
    "01110100010100011010", "00011100100010000111",
)


def _reference():
    return np.array([[int(v) for v in row] for row in REFERENCE_ROWS], dtype=np.int8)


def _transform(grid, index):
    """0..3: CCW rotations; 4..7: left/right reflection then CCW rotation."""
    return np.rot90(np.fliplr(grid) if index >= 4 else grid, index % 4).copy()


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30 and colors 0..9")
    return array.astype(np.int8, copy=False)


def _locations(full, patch):
    """Exact matching, pruning by the first row before checking later rows."""
    h, w = patch.shape
    if h > 20 or w > 20:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    rr, cc = np.nonzero(np.all(windows[:, :, 0, :] == patch[0], axis=-1))
    for r in range(1, h):
        if not len(rr):
            break
        keep = np.all(full[rr[:, None] + r, cc[:, None] + np.arange(w)] == patch[r], axis=1)
        rr, cc = rr[keep], cc[keep]
    return [(int(r), int(c)) for r, c in zip(rr, cc)]


def _matches(grid):
    colors = np.unique(grid)
    if len(colors) != 2:
        return []
    patch = (grid == colors[1]).astype(np.int8)
    reference = _reference()
    return [(i, swap, r, c)
            for i in range(8)
            for swap in range(2)
            for r, c in _locations(_transform(reference, i), patch if not swap else 1 - patch)]


def _completion(grid, match):
    i, swap, _, _ = match
    colors = np.unique(grid)
    labels = _transform(_reference(), i)
    return colors[1 - labels if swap else labels].astype(np.int8)


def _recover(grid):
    matches = _matches(grid)
    completions = {_key(_completion(grid, match)): match for match in matches}
    if len(completions) != 1:
        raise ValueError("input must uniquely identify a full reference, orientation and color map")
    return _completion(grid, next(iter(completions.values())))


def _parity_corner(full):
    # Keep the causal order explicit. The ORIGINAL crop's dimensions are irrelevant.
    if full.shape[0] != full.shape[1] or full.shape[0] < 2:
        raise ValueError("corner selection requires a square of side at least two")
    return (full[:2, :2] if full.shape[0] % 2 else full[:2, -2:]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only exact recognition, completion, then completed-side parity crop.

No generation call, seed, demonstrations or mutable cache is consulted. The
fixed family bitmap above is the only prior. Ambiguous completions are rejected,
even when their small final answers happen to agree.
    """
    return _parity_corner(_recover(_grid(input_grid)))


def _integer(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    return int(value)


def _visible_corner(match, shape):
    _, _, r, c = match
    h, w = shape
    return r == 0 and h >= 2 and c <= 18 and c + w == 20


def _anchors(rng, full):
    """Two opposite overlapping crops reveal all 400 cells in their union."""
    candidates = []
    target = _parity_corner(full)
    for axis in range(2):
        for loss in range(1, 7):
            first = full[:, :-loss] if axis else full[:-loss, :]
            second = full[:, loss:] if axis else full[loss:, :]
            if (len(_matches(first)) == len(_matches(second)) == 1
                    and any(not np.array_equal(target, p[:2, -2:]) for p in (first, second))):
                candidates.append((first.copy(), second.copy()))
    if not candidates:
        raise RuntimeError("fixed reference has no valid overlapping teaching crops")
    chosen = candidates[int(rng.integers(len(candidates)))]
    return list(chosen if rng.random() < .5 else chosen[::-1])


def _sample_pair(rng, full, used, test):
    def accept(h, w, r, c):
        patch = full[r:r + h, c:c + w].copy()
        if _key(patch) in used or len(np.unique(patch)) != 2:
            return None
        matches = _matches(patch)
        if len(matches) != 1 or (test and _visible_corner(matches[0], patch.shape)):
            return None
        # Verify the intended full reconstruction, not merely the tiny output.
        if not np.array_equal(_completion(patch, matches[0]), full):
            return None
        used.add(_key(patch))
        return {"input": patch, "output": _parity_corner(full)}

    for _ in range(192):
        mode = int(rng.integers(4))
        if mode == 0:
            h, w = int(rng.integers(4, 11)), int(rng.integers(4, 11))
        elif mode == 1:
            h, w = int(rng.integers(2, 5)), int(rng.integers(12, 21))
            if rng.random() < .5:
                h, w = w, h
        elif mode == 2:
            h, w = int(rng.integers(7, 13)), int(rng.integers(7, 13))
        elif test:
            h, w = int(rng.integers(3, 13)), int(rng.integers(3, 13))
        else:
            h, w = int(rng.integers(12, 20)), int(rng.integers(12, 20))
        r, c = int(rng.integers(21 - h)), int(rng.integers(21 - w))
        pair = accept(h, w, r, c)
        if pair is not None:
            return pair
    # Finite, deterministic valid-family fallback, at most 400 candidates.
    # Every fallback is < half the template and omits its top row.
    for h in range(8, 13):
        for w in range(8, 13):
            for r in range(1, 5):
                for c in range(4):
                    pair = accept(h, w, r, c)
                    if pair is not None:
                        return pair
    raise RuntimeError("bounded crop search exhausted")


def _rotation_indices(base):
    reference = _reference()
    orbit = [_transform(reference, i) for i in range(8)]
    return [next(i for i, a in enumerate(orbit) if np.array_equal(a, np.rot90(orbit[base], k)))
            for k in range(4)]


def _compose_orientations(after, before):
    """Compose R^r F^s transforms, where F reflects left/right."""
    ar, af = after % 4, after // 4
    br, bf = before % 4, before // 4
    return (ar + (-br if af else br)) % 4 + 4 * (af ^ bf)


def _supported_orientations(base, observed):
    """The subgroup generated by transformations visible relative to base."""
    inverse = (-base) % 4 if base < 4 else base
    group = {0} | {_compose_orientations(i, inverse) for i in observed}
    while True:
        expanded = group | {_compose_orientations(a, b) for a in group for b in group}
        if expanded == group:
            return {_compose_orientations(i, base) for i in group}
        group = expanded


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    base = int(rng.integers(8))
    colors = rng.choice(10, size=2, replace=False).astype(np.int8)
    full = colors[_transform(_reference(), base)]
    puzzle, used = {"train": [], "test": []}, set()
    for patch in _anchors(rng, full):
        used.add(_key(patch))
        puzzle["train"].append({"input": patch, "output": _parity_corner(full)})

    rotations = _rotation_indices(base)
    mirror = next(i for i in range(8) if np.array_equal(
        _transform(_reference(), i), np.fliplr(_transform(_reference(), base))))
    train_orientations = [rotations[1], mirror]
    while len(train_orientations) < nt - 2:
        train_orientations.append(int(rng.integers(8)))
    for j, i in enumerate(train_orientations[:nt - 2]):
        if j == 0:
            # Explicit recoloring evidence before any test can use a new palette.
            choices = [(a, b) for a in range(10) for b in range(a + 1, 10)
                       if {a, b} != set(colors)]
            palette = np.array(choices[int(rng.integers(len(choices)))], dtype=np.int8)
            if rng.random() < .5:
                palette = palette[::-1]
        else:
            palette = rng.choice(10, 2, replace=False).astype(np.int8)
        candidate = palette[_transform(_reference(), i)]
        puzzle["train"].append(_sample_pair(rng, candidate, used, False))

    test_pool = [base] if nt == 2 else rotations if nt == 3 else list(range(8))
    for _ in range(nq):
        i = int(rng.choice(test_pool))
        palette = colors if nt == 2 else rng.choice(10, 2, replace=False).astype(np.int8)
        candidate = palette[_transform(_reference(), i)]
        puzzle["test"].append(_sample_pair(rng, candidate, used, True))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, train_recoveries, test_recoveries = set(), [], []
    local_failures = wrong_left = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            _require(min(inp.shape) >= 2 and inp.size < 400 and len(np.unique(inp)) == 2,
                     "need a proper binary crop with both sides at least two")
            matches = _matches(inp)
            _require(len(matches) == 1, "crop must identify orientation, roles and location uniquely")
            full = _completion(inp, matches[0])
            _require(full.shape == (20, 20), "the saved proposal requires 20x20")
            _require(out.shape == (2, 2) and np.array_equal(out, full[:2, -2:]), "incorrect corner")
            _require(np.array_equal(solve(inp), out), "input-only solver disagrees")
            _require(set(np.unique(out)) <= set(np.unique(inp)), "corner colors must be copied")
            if split == "test":
                _require(inp.size <= 144, "tests use smaller fragments or narrow strips")
                _require(not _visible_corner(matches[0], inp.shape), "test must require missing corner cells")
                test_recoveries.append((full, matches[0], inp))
            else:
                train_recoveries.append((full, matches[0], inp))
                local_failures += not np.array_equal(out, inp[:2, -2:])
                wrong_left += not np.array_equal(out, full[:2, :2])
    # The first two inputs visibly teach the whole fixed image by overlap.
    first, second = train_recoveries[:2]
    _require(np.array_equal(first[0], second[0]), "teaching crops must share one colored orientation")
    cover = np.zeros((20, 20), dtype=bool)
    for full, match, inp in (first, second):
        _, _, r, c = match
        h, w = inp.shape
        _require(inp.size >= 280, "teaching crops must be large")
        cover[r:r + h, c:c + w] = True
    _require(bool(cover.all()), "teaching inputs must jointly expose the full reference")
    _require(local_failures > 0 and wrong_left > 0, "training must refute direct crop and wrong corner")
    supported = _supported_orientations(first[1][0], {m[0] for _, m, _ in train_recoveries})
    _require(all(match[0] in supported for _, match, _ in test_recoveries),
             "test geometry needs demonstrated transformations")
    if nt == 2:
        _require(all(np.array_equal(full, first[0]) for full, _, _ in test_recoveries),
                 "minimal teaching set keeps test geometry and palette familiar")
    else:
        _require(len({tuple(np.unique(inp)) for _, _, inp in train_recoveries}) >= 2,
                 "novel test palettes need recoloring evidence")


def _signature(puzzle):
    return tuple((s, _key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """42 streamed seeds, all legal counts, corruption and ambiguity checks.

    Retains only small feature sets and one puzzle at a time. Returns metrics;
    does not save the reconstructible corpus. Parity necessity is reported as
    zero rather than asserted to have been established.
    """
    signatures, shapes, offsets, orientations, palettes, structures = (set() for _ in range(6))
    local_wrong = input_parity_wrong = left_wrong = uniform = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                match = _matches(inp)[0]
                full = _completion(inp, match)
                shapes.add(inp.shape)
                offsets.add(match[2:])
                orientations.add(match[0])
                palettes.add(tuple(int(x) for x in np.unique(inp)))
                structures.add(_key((inp == inp[0, 0]).astype(np.int8)))
                pairs += 1
                local_wrong += not np.array_equal(out, inp[:2, -2:])
                input_choice = inp[:2, :2] if inp.shape[0] % 2 else inp[:2, -2:]
                input_parity_wrong += not np.array_equal(out, input_choice)
                left_wrong += not np.array_equal(out, full[:2, :2])
                uniform += len(np.unique(out)) == 1
                _require(np.array_equal(out, full[:2, -2:]), "always-right equivalence must be recorded")
    _require(len(signatures) == 42 and len(shapes) >= 40 and len(offsets) >= 30, "insufficient crop variation")
    _require(orientations == set(range(8)) and len(palettes) >= 30, "orientation/palette variation")
    _require(len(structures) >= 120 and local_wrong >= 42 and left_wrong == pairs, "structural/witness checks")
    _require(input_parity_wrong >= 42 and uniform > 0, "crop-parity and uniform-corner coverage")
    _require(_signature(generate_puzzle(0)) == _signature(generate_puzzle(0)), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1234 + nt * 10 + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "count mismatch")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for invalid in (_reference()[:4, :4], np.ones((3, 3), dtype=int)):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or one-color crop accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Correct labels alone do not establish an unseen geometric operation.
    # Remove the geometric teaching evidence while retaining new palettes,
    # distinct exact crops, overlapping anchors and correct corner answers.
    bad = generate_puzzle(0)
    full = _recover(bad["train"][0]["input"])
    used = {_key(p["input"]) for p in bad["train"][:2]}
    recolored = np.array([1, 2], dtype=np.int8)[(full == full[0, 0]).astype(np.int8)]
    rng = np.random.default_rng(912345)
    for i in range(2, len(bad["train"])):
        bad["train"][i] = _sample_pair(rng, recolored, used, False)
    bad["test"][0] = _sample_pair(rng, _transform(full, 1), used, True)
    try:
        validate_generated_puzzle(bad)
    except AssertionError as error:
        _require(str(error) == "test geometry needs demonstrated transformations",
                 "teaching-evidence regression failed for an unrelated reason")
    else:
        raise AssertionError("unsupported test geometry accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "crop_shapes": len(shapes), "offsets": len(offsets),
            "orientations": len(orientations), "palettes": len(palettes),
            "normalized_input_structures": len(structures),
            "wrong_direct_right_crop": local_wrong, "wrong_input_height_parity": input_parity_wrong,
            "wrong_completed_left_corner": left_wrong, "uniform_outputs": uniform,
            "unsupported_test_geometry_rejected": True,
            "parity_necessity_witnesses": 0,
            "limitation": "fixed even reference makes always-right indistinguishable from parity"}
