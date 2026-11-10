"""Compositional enlarged-stencil / compact-payload ARC puzzles."""

import hashlib
import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "49e0ce48a6695185adf852a3ded3090778c1faf75bb962615a2ddace35370dbe",
    "ea2e500b3a22aa022a7fb3f20ac2a5bf48a06ff1c7d4ab137cb83874fcaf2cfb",
    "236224fadfbb6cb2bdd3347cc42309d29fa5e3538edb21925f5c7f0984127056",
    "c830b43c461af109ea290610c68ec6b4eeb9ed990d70eaf49e9d90bc2989e062",
}
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 2), (8, 3), (9, 4), (10, 1))
_SPECS = tuple((n, k) for n in range(2, 7) for k in range(2, 7)
               if n * k + n <= 27 and (n != 2 or k >= 3))
_LAYOUTS = ("below", "right", "above", "left", "diagonal", "overlap")


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    a = _checked_count(num_train, "num_train", 2, 10)
    b = _checked_count(num_test, "num_test", 1, 4)
    if a is None and b is None:
        a, b = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _dihedral(mask):
    for basis in (mask, np.fliplr(mask)):
        for turns in range(4):
            yield np.rot90(basis, turns)


def _symmetric(mask):
    return any(np.array_equal(mask, image) for image in list(_dihedral(mask))[1:])


def _sample_mask(rng, n, asymmetric=False, blank_corner=False):
    for _ in range(200):
        motif = int(rng.integers(0, 11))
        mask = np.zeros((n, n), dtype=bool)
        if motif <= 2:
            mask = rng.random((n, n)) < float(rng.uniform(0.18, 0.87))
        elif motif == 3:  # cross
            center = n // 2
            mask[center, :] = True
            mask[:, center] = True
        elif motif == 4:  # bars / short branches
            mask[int(rng.integers(n)), :] = True
            mask[:, int(rng.integers(n))] = True
            if n > 3 and bool(rng.integers(2)):
                mask[int(rng.integers(n)), :] = True
        elif motif == 5:  # frame with one or more holes
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
            if n > 3:
                mask[1:-1, 1:-1] = rng.random((n - 2, n - 2)) < 0.25
        elif motif == 6:  # corners and occasional interior islands
            mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = True
            if n > 2:
                mask[1:-1, 1:-1] = rng.random((n - 2, n - 2)) < 0.35
        elif motif == 7:  # checkerboard islands
            mask = np.indices((n, n)).sum(axis=0) % 2 == int(rng.integers(2))
        elif motif == 8:  # crossing diagonals
            indices = np.arange(n)
            mask[indices, indices] = True
            mask[indices, n - 1 - indices] = True
        elif motif == 9:  # sparse islands that still expose every edge
            mask[0, 0] = mask[-1, -1] = True
            if n > 3 and bool(rng.integers(2)):
                mask[int(rng.integers(1, n - 1)),
                     int(rng.integers(1, n - 1))] = True
        else:  # nearly full, with a few unmistakable empty cells
            mask[:] = True
            for _ in range(int(rng.integers(1, min(4, n * n - 1)))):
                mask[int(rng.integers(n)), int(rng.integers(n))] = False
        mask = list(_dihedral(mask))[int(rng.integers(8))].copy()
        occupied = int(mask.sum())
        if not 2 <= occupied <= n * n - 1:
            continue
        if not (mask[0].any() and mask[-1].any() and
                mask[:, 0].any() and mask[:, -1].any()):
            continue
        if asymmetric and _symmetric(mask):
            continue
        if blank_corner and all(mask[r, c] for r in (0, -1) for c in (0, -1)):
            continue
        return mask
    # Random masks make the constrained n=3 witness deterministic even in an
    # extremely unlucky stream, without accepting an ambiguous all-full mask.
    mask = np.zeros((n, n), dtype=bool)
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = True
    mask[1, 1] = True
    if blank_corner:
        mask[-1, -1] = False
    if asymmetric and _symmetric(mask):
        mask[1, 0] = True
    assert not asymmetric or not _symmetric(mask)
    return mask


def _sample_payload(rng, n, background, carrier, mask, overlap):
    colors = np.array([c for c in range(10) if c != background], dtype=np.int8)
    others = colors[colors != carrier]
    maximum = min(n * n, 9 if overlap else 8)
    if maximum >= 7 and rng.random() < 0.48:
        size = int(rng.integers(7, min(8, maximum) + 1))
    else:
        size = int(rng.integers(3, min(6, maximum) + 1))
    if overlap:
        palette = np.concatenate((np.array([carrier], dtype=np.int8),
                                  rng.choice(others, size - 1, replace=False)))
    else:
        palette = rng.choice(others, size, replace=False)
    for _ in range(100):
        values = rng.choice(palette, n * n).astype(np.int8)
        values[:size] = palette
        values = values[rng.permutation(n * n)].reshape(n, n)
        noncarrier = values != carrier
        if not (noncarrier[0].any() and noncarrier[-1].any() and
                noncarrier[:, 0].any() and noncarrier[:, -1].any()):
            continue
        if np.unique(values[mask]).size < 2:
            continue
        return values
    # A shuffled palette covers every color, and the checkerboard order keeps
    # the noncarrier part of each edge visible for structural recovery.
    values = np.resize(palette, n * n).reshape(n, n).copy()
    if not (np.any(values[0] != carrier) and np.any(values[-1] != carrier)
            and np.any(values[:, 0] != carrier) and np.any(values[:, -1] != carrier)
            and np.unique(values[mask]).size >= 2):
        raise RuntimeError("payload sampling failed")
    return values


def _placement(rng, side, n, scale, mask, layout):
    if layout in ("below", "above", "right", "left"):
        gap = int(rng.integers(0, min(4, 30 - side - n - 2) + 1))
        cross = int(rng.integers(-n + 1, side + 1))
        if layout == "below":
            return side + gap, cross
        if layout == "above":
            return -n - gap, cross
        if layout == "right":
            return cross, side + gap
        return cross, -n - gap
    if layout == "diagonal":
        gap_r = int(rng.integers(0, min(3, 30 - side - n - 2) + 1))
        gap_c = int(rng.integers(0, min(3, 30 - side - n - 2) + 1))
        return (side + gap_r if bool(rng.integers(2)) else -n - gap_r,
                side + gap_c if bool(rng.integers(2)) else -n - gap_c)
    corners = [(r, c) for r in (0, -1) for c in (0, -1) if not mask[r, c]]
    if not corners:
        raise RuntimeError("overlap needs an empty corner")
    r, c = corners[int(rng.integers(len(corners)))]
    depth = int(rng.integers(1, min(n - 1, scale) + 1))
    return ((-n + depth if r == 0 else side - depth),
            (-n + depth if c == 0 else side - depth))


def _assemble(rng, stencil, payload, background, mask, layout, scale):
    side, n = len(stencil), len(payload)
    dr, dc = _placement(rng, side, n, scale, mask, layout)
    row_low, col_low = min(0, dr), min(0, dc)
    row_high, col_high = max(side, dr + n), max(side, dc + n)
    base_h, base_w = row_high - row_low, col_high - col_low
    if base_h > 28 or base_w > 28:
        raise RuntimeError("placement exceeds canvas")
    # Make the local background the dominant color, including dense masks.
    carrier_cells = int(np.count_nonzero(stencil))
    min_area = 2 * carrier_cells + n * n + 1
    possibilities = [(h, w) for h in range(base_h + 2, 31)
                     for w in range(base_w + 2, 31) if h * w >= min_area]
    if not possibilities:
        raise RuntimeError("cannot show background clearly")
    h, w = possibilities[int(rng.integers(len(possibilities)))]
    before_r = int(rng.integers(1, h - base_h))
    before_c = int(rng.integers(1, w - base_w))
    sr, sc = before_r - row_low, before_c - col_low
    pr, pc = sr + dr, sc + dc
    canvas = np.full((h, w), background, dtype=np.int8)
    canvas[sr:sr + side, sc:sc + side] = stencil
    patch = canvas[pr:pr + n, pc:pc + n]
    if np.any(patch != background):
        raise RuntimeError("payload would overwrite a filled block")
    canvas[pr:pr + n, pc:pc + n] = payload
    return canvas


def _array_key(grid):
    return (tuple(grid.shape), grid.tobytes())


def _pair_digest(inp, out):
    raw = "|".join(f"{len(g)}x{len(g[0])}:" + "".join(str(int(v)) for v in g.flat)
                   for g in (inp, out))
    return hashlib.sha256(raw.encode()).hexdigest()


def _recover_pair_structure(inp, out=None):
    h, w = inp.shape
    background = int(inp[0, 0])
    assert all(int(inp[r, c]) == background for r in (0, -1) for c in (0, -1))
    foreground = np.unique(inp[inp != background])
    explanations = []
    for carrier_value in foreground:
        carrier = int(carrier_value)
        other_positions = np.argwhere((inp != background) & (inp != carrier))
        if not len(other_positions):
            continue
        p0 = other_positions.min(axis=0)
        p1 = other_positions.max(axis=0) + 1
        n = int(p1[0] - p0[0])
        if not (2 <= n <= 6 and int(p1[1] - p0[1]) == n):
            continue
        payload = inp[p0[0]:p1[0], p0[1]:p1[1]].copy()
        if np.any(payload == background) or np.unique(payload).size < 3:
            continue
        residue = inp.copy()
        residue[p0[0]:p1[0], p0[1]:p1[1]] = background
        if np.any((residue != background) & (residue != carrier)):
            continue
        positions = np.argwhere(residue == carrier)
        if not len(positions):
            continue
        s0 = positions.min(axis=0)
        s1 = positions.max(axis=0) + 1
        side = int(s1[0] - s0[0])
        if side != int(s1[1] - s0[1]) or side % n:
            continue
        scale = side // n
        if scale < 2 or scale > 6:
            continue
        mask = np.zeros((n, n), dtype=bool)
        okay = True
        for r in range(n):
            for c in range(n):
                block = residue[s0[0] + r * scale:s0[0] + (r + 1) * scale,
                                s0[1] + c * scale:s0[1] + (c + 1) * scale]
                if np.all(block == carrier):
                    mask[r, c] = True
                elif not np.all(block == background):
                    okay = False
                    break
            if not okay:
                break
        if not okay or not (mask[0].any() and mask[-1].any() and
                           mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not (2 <= int(mask.sum()) < n * n):
            continue
        reconstruction = np.kron(mask.astype(np.int8),
                                 np.ones((scale, scale), dtype=np.int8))
        reconstruction = np.where(reconstruction, carrier, background)
        if not np.array_equal(residue[s0[0]:s1[0], s0[1]:s1[1]], reconstruction):
            continue
        expected = np.where(mask, payload, background).astype(np.int8)
        if out is not None and (out.shape != (n, n) or not np.array_equal(out, expected)):
            continue
        vertical = ("above" if p1[0] <= s0[0] else
                    "below" if p0[0] >= s1[0] else "middle")
        horizontal = ("left" if p1[1] <= s0[1] else
                      "right" if p0[1] >= s1[1] else "middle")
        if vertical != "middle" and horizontal != "middle":
            layout = "diagonal"
        elif vertical != "middle":
            layout = vertical
        elif horizontal != "middle":
            layout = horizontal
        else:
            layout = "overlap"
        explanations.append({"n": n, "scale": scale, "background": background,
                             "carrier": carrier, "payload": payload, "mask": mask,
                             "layout": layout, "overlap": bool(np.any(payload == carrier)),
                             "palette_size": int(np.unique(payload).size),
                             "expected_output": expected,
                             "stencil_box": (tuple(s0), tuple(s1)),
                             "payload_box": (tuple(p0), tuple(p1))})
    assert len(explanations) == 1, "input must have one recoverable stencil and payload"
    return explanations[0]


def _choose_train_specs(rng, count):
    first = _SPECS[int(rng.integers(len(_SPECS)))]
    while first[0] == 2:
        first = _SPECS[int(rng.integers(len(_SPECS)))]
    candidates = [spec for spec in _SPECS
                  if spec[0] != first[0] and spec[1] != first[1]]
    second = candidates[int(rng.integers(len(candidates)))]
    specs = [first, second]
    while len(specs) < count:
        specs.append(_SPECS[int(rng.integers(len(_SPECS)))])
    return specs


def _choose_test_spec(rng, train_specs):
    roll = rng.random()
    if roll < 0.28:
        return train_specs[int(rng.integers(len(train_specs)))]
    recombinations = [spec for spec in _SPECS if spec not in train_specs
                      and spec[0] in {n for n, _ in train_specs}
                      and spec[1] in {k for _, k in train_specs}]
    if roll < 0.72 and recombinations:
        return recombinations[int(rng.integers(len(recombinations)))]
    fresh = [spec for spec in _SPECS if spec not in train_specs
             and (spec[0] not in {n for n, _ in train_specs}
                  or spec[1] not in {k for _, k in train_specs})]
    if fresh:
        return fresh[int(rng.integers(len(fresh)))]
    return _SPECS[int(rng.integers(len(_SPECS)))]


def _make_pair(rng, n, scale, background, carrier, layout, overlap, asymmetric):
    for _ in range(100):
        mask = _sample_mask(rng, n, asymmetric=asymmetric,
                            blank_corner=(layout == "overlap"))
        payload = _sample_payload(rng, n, background, carrier, mask, overlap)
        stencil = np.where(np.kron(mask, np.ones((scale, scale), dtype=bool)),
                           carrier, background).astype(np.int8)
        try:
            inp = _assemble(rng, stencil, payload, background, mask, layout, scale)
            out = np.where(mask, payload, background).astype(np.int8)
            found = _recover_pair_structure(inp, out)
            if (found["n"], found["scale"], found["background"], found["carrier"]) != (n, scale, background, carrier):
                continue
            if _pair_digest(inp, out) in _SOURCE_PAIR_DIGESTS:
                continue
            return {"input": inp, "output": out}
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("could not construct an unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    # Direct row/column alignment is the common rule. The only optional shared
    # convention is whether a puzzle uses one background or local backgrounds.
    varying_background = bool(rng.integers(2))
    background_cycle = rng.permutation(np.arange(10)).tolist()
    if not varying_background:
        background_cycle = [int(rng.choice([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]
    train_specs = _choose_train_specs(rng, train_count)
    test_specs = [_choose_test_spec(rng, train_specs) for _ in range(test_count)]
    layouts = [str(rng.choice(_LAYOUTS)) for _ in range(train_count + test_count)]
    if rng.random() < 0.3:
        layouts[1] = layouts[0]
    first_overlap = bool(rng.integers(2))
    train, test, seen = [], [], set()
    for index, (n, scale) in enumerate(train_specs + test_specs):
        background = int(background_cycle[index % len(background_cycle)])
        carrier = int(rng.choice([c for c in range(10) if c != background]))
        # The first two demonstrations deliberately show both local palette
        # modes; later choices are pair-level and independent of background.
        overlap = (first_overlap if index == 0 else
                   not first_overlap if index == 1 else bool(rng.integers(2)))
        layout = layouts[index]
        for _ in range(40):
            pair = _make_pair(rng, n, scale, background, carrier,
                              layout, overlap, asymmetric=(index == 0))
            key = _array_key(pair["input"])
            if key not in seen:
                seen.add(key)
                (train if index < train_count else test).append(pair)
                break
        else:
            raise RuntimeError("could not make a unique input")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = (_array_key(pair["input"]), _array_key(pair["output"]))
            assert key[0] not in seen_inputs and key not in seen_pairs
            assert _pair_digest(pair["input"], pair["output"]) not in _SOURCE_PAIR_DIGESTS
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            assert not np.array_equal(pair["input"], pair["output"])
            analysis = _recover_pair_structure(pair["input"], pair["output"])
            assert (analysis["n"], analysis["scale"]) in _SPECS
            assert np.unique(analysis["payload"][analysis["mask"]]).size >= 2
            analyses[split].append(analysis)
    train = analyses["train"]
    all_analyses = train + analyses["test"]
    assert not _symmetric(train[0]["mask"]), "training needs an alignment witness"
    assert {a["overlap"] for a in train} == {False, True}
    train_backgrounds = {a["background"] for a in train}
    if len(train_backgrounds) == 1:
        assert {a["background"] for a in all_analyses} == train_backgrounds
    assert len({(a["n"], a["scale"]) for a in train}) >= 2
    assert all(np.array_equal(a["expected_output"] != a["background"], a["mask"])
               for a in all_analyses)


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, keys = set(), set(), set()
    specs, masks, layouts, backgrounds, palettes = set(), set(), set(), set(), set()
    symmetric = sparse = dense = overlap_count = 0
    within_spec = within_layout = within_background = 0
    novel_test = recombined_test = familiar_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        keys.add(_puzzle_key(puzzle))
        train_analyses = [_recover_pair_structure(p["input"], p["output"]) for p in puzzle["train"]]
        test_analyses = [_recover_pair_structure(p["input"], p["output"]) for p in puzzle["test"]]
        all_analyses = train_analyses + test_analyses
        within_spec += len({(a["n"], a["scale"]) for a in all_analyses}) >= 2
        within_layout += len({a["layout"] for a in all_analyses}) >= 2
        within_background += len({a["background"] for a in all_analyses}) >= 2
        train_specs = {(a["n"], a["scale"]) for a in train_analyses}
        ns, scales = {a["n"] for a in train_analyses}, {a["scale"] for a in train_analyses}
        for a in test_analyses:
            spec = a["n"], a["scale"]
            if spec in train_specs:
                familiar_test += 1
            elif a["n"] in ns and a["scale"] in scales:
                recombined_test += 1
            else:
                novel_test += 1
        for a in all_analyses:
            specs.add((a["n"], a["scale"]))
            masks.add((a["n"], tuple(a["mask"].flat)))
            layouts.add(a["layout"])
            backgrounds.add(a["background"])
            palettes.add(a["palette_size"])
            symmetric += _symmetric(a["mask"])
            sparse += a["mask"].mean() <= 0.3
            dense += a["mask"].mean() >= 0.7
            overlap_count += a["overlap"]
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(keys) == 42
    assert len(specs) >= 12 and len(masks) >= 50
    assert layouts == set(_LAYOUTS) and len(backgrounds) == 10
    assert max(palettes) >= 8 and min(palettes) <= 4
    assert symmetric >= 10 and sparse >= 3 and dense >= 10 and overlap_count >= 20
    assert within_spec >= 38 and within_layout >= 20 and within_background >= 10
    assert familiar_test >= 5 and recombined_test >= 5 and novel_test >= 5
    for index, (a, b) in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(10000 + index, a, b)
        validate_generated_puzzle(puzzle)
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
    explicit = generate_puzzle(9001, 6, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(9001, 6, 3))
    assert _puzzle_key(explicit) != _puzzle_key(generate_puzzle(9002, 6, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(12000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
