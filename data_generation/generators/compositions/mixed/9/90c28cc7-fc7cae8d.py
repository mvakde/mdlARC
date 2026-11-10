"""Guide-bounded, independently expanded mosaics with identity-keyed orientation.

Only visible input pixels determine solving. No parent code or files are loaded.
Generation uses named D4 array operations; checking/solving moves coordinates
using side normals. Nothing is written to disk by this module.
"""
from __future__ import annotations

import hashlib
import numpy as np

# Side numbers run clockwise: top, right, bottom, left.
_NORMALS = np.array(((-1, 0), (0, 1), (1, 0), (0, -1)), dtype=int)
_NAMES = ("identity", "ccw90", "half_turn", "cw90", "mirror_lr",
          "main_diagonal", "mirror_ud", "anti_diagonal")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer rectangles with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _side(a, side):
    return (a[0], a[:, -1], a[-1], a[:, 0])[side]


def _side_map(code, side):
    return ((side if code < 4 else -side) - code % 4) % 4


def _transform(a, code):
    return np.rot90(a if code < 4 else np.fliplr(a), code % 4).copy()


def _asymmetric(a):
    return len({_key(_transform(a, k)) for k in range(8)}) == 8


def _coordinate_orient(a, guides, controls):
    """Build the signed permutation from two matched outward side normals."""
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in guides.items():
        target = next(s for s, c in controls.items() if c == color)
        matrix += np.outer(_NORMALS[target], _NORMALS[side])
    assert np.array_equal(matrix.T @ matrix, np.eye(2, dtype=int))
    positions = matrix @ np.indices(a.shape).reshape(2, -1)
    positions -= positions.min(axis=1, keepdims=True)
    result = np.empty(tuple(positions.max(axis=1) + 1), dtype=np.int8)
    result[positions[0], positions[1]] = a.ravel()
    return result


def _compress(a):
    """Retain the first element of each MAXIMAL consecutive equal-vector run."""
    rows = np.r_[0, np.flatnonzero(np.any(a[1:] != a[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0)) + 1]
    logical = a[np.ix_(rows, cols)].copy()
    heights = np.diff(np.r_[rows, a.shape[0]])
    widths = np.diff(np.r_[cols, a.shape[1]])
    return logical, heights, widths


def _visible(a):
    return (np.all(np.any(a[1:] != a[:-1], axis=1))
            and np.all(np.any(a[:, 1:] != a[:, :-1], axis=0)))


def _single_seam(a):
    return (np.any(np.sum(a[1:] != a[:-1], axis=1) == 1)
            or np.any(np.sum(a[:, 1:] != a[:, :-1], axis=0) == 1))


def _separated_repeat(a):
    return any(np.array_equal(b[i], b[j]) for b in (a, a.T)
               for i in range(len(b)) for j in range(i + 2, len(b)))


def _blank_sides(a, background):
    return tuple(s for s in range(4) if np.all(_side(a, s) == background))


def _paint_guide(a, bounds, side, color):
    r, c, h, w = bounds
    if side == 0:
        a[r - 1, c:c + w] = color
    elif side == 1:
        a[r:r + h, c + w] = color
    elif side == 2:
        a[r + h, c:c + w] = color
    else:
        a[r:r + h, c - 1] = color


def _bar(a, color):
    # Controls occupy only the outermost border. Each identity has exactly
    # one uninterrupted interior bar, separate from every other marker cell.
    rr, cc = np.where(a[1:-1, 1:-1] == color)
    rr, cc = rr + 1, cc + 1
    assert len(rr) >= 2, "missing guide"
    if np.all(rr == rr[0]):
        assert np.array_equal(np.sort(cc), np.arange(cc.min(), cc.max() + 1))
        return "H", int(rr[0]), int(cc.min()), int(cc.max())
    assert np.all(cc == cc[0]), "guide must be straight and one cell wide"
    assert np.array_equal(np.sort(rr), np.arange(rr.min(), rr.max() + 1))
    return "V", int(cc[0]), int(rr.min()), int(rr.max())


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(a[p] == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    controls = {}
    for side in range(4):
        core = _side(a, side)[1:-1]
        if len(core) >= 2 and np.all(core == core[0]) and int(core[0]) != bg:
            controls[side] = int(core[0])
    assert len(controls) == 2 and len(set(controls.values())) == 2
    sides = sorted(controls)
    assert (sides[1] - sides[0]) % 4 in (1, 3), "controls must be adjacent"
    bars = [(_bar(a, color), color) for color in controls.values()]
    horizontal = [(bar, color) for bar, color in bars if bar[0] == "H"]
    vertical = [(bar, color) for bar, color in bars if bar[0] == "V"]
    assert len(horizontal) == len(vertical) == 1
    (_, hr, c, last_c), hc = horizontal[0]
    (_, vc, r, last_r), vcolor = vertical[0]
    h, w = last_r - r + 1, last_c - c + 1
    assert hr in (r - 1, r + h) and vc in (c - 1, c + w)
    guides = {0 if hr == r - 1 else 2: hc, 3 if vc == c - 1 else 1: vcolor}
    bounds = r, c, h, w
    r0, r1 = r - (0 in guides), r + h + (2 in guides)
    c0, c1 = c - (3 in guides), c + w + (1 in guides)
    assert r0 >= 2 and c0 >= 2 and r1 <= a.shape[0] - 2 and c1 <= a.shape[1] - 2
    payload = a[r:r + h, c:c + w].copy()
    assert not set(map(int, np.unique(payload))) & set(controls.values())
    # The exact interior is payload + two bars + uniform background. This
    # verifies the moat and rejects extra potential crops/marker fragments.
    expected = np.full(a.shape, bg, dtype=np.int8)
    expected[r:r + h, c:c + w] = payload
    for side, color in guides.items():
        _paint_guide(expected, bounds, side, color)
    assert np.array_equal(expected[1:-1, 1:-1], a[1:-1, 1:-1])
    noise = set()
    for side in set(range(4)) - set(controls):
        run, previous = 0, None
        for x in _side(a, side)[1:-1]:
            x = int(x)
            if x == bg:
                run, previous = 0, None
                continue
            noise.add(x)
            run = run + 1 if previous == x else 1
            previous = x
            assert run <= 2, "unused borders must remain fragmented"
    assert len(noise) <= 1
    assert not noise & (set(map(int, np.unique(payload))) | set(controls.values()) | {bg})
    codes = [k for k in range(8) if all(controls.get(_side_map(k, s)) == color
                                      for s, color in guides.items())]
    assert len(codes) == 1
    logical, rows, cols = _compress(payload)
    return dict(background=bg, controls=controls, guides=guides, bounds=bounds,
                payload=payload, logical=logical, rows=rows, cols=cols, code=codes[0])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Crop by guides, collapse complete row/column runs, then align identities."""
    info = _decode(input_grid)
    return _coordinate_orient(info["logical"], info["guides"], info["controls"])


def _palette(rng):
    colors = [int(x) for x in rng.permutation(10)]
    return dict(bg=colors[0], markers=colors[1:3], noise=colors[3],
                content=colors[4:4 + int(rng.integers(2, 7))])


def _teaching_logical(rng, palette, max_dim):
    """A variable motif with a blank edge, one-pixel seam and separated repeat."""
    colors, bg = palette["content"], palette["bg"]
    for _ in range(256):
        h, w = int(rng.integers(4, max_dim + 1)), int(rng.integers(3, max_dim + 1))
        a = rng.choice(colors, (h, w)).astype(np.int8)
        a[0] = bg
        start = int(rng.integers(1, h - 2))
        j = int(rng.integers(w - 1))
        a[start, j + 1] = a[start, j]
        a[start + 1] = a[start]
        a[start + 2] = a[start]
        # Mutate one of the equal neighbors, exposing that global seam.
        col = j + int(rng.integers(2))
        a[start + 1, col] = int(rng.choice([c for c in colors if c != a[start, col]]))
        a = np.rot90(a, int(rng.integers(4))).copy()
        if _visible(a) and _asymmetric(a):
            return a
    raise RuntimeError("bounded teaching-motif sampling exhausted")


def _logical(rng, palette, shape):
    colors = palette["content"]
    for _ in range(256):
        h, w = shape
        style = int(rng.integers(4))
        if style == 0:  # General small mosaics, no fixed corner anchors.
            a = rng.choice(colors, shape).astype(np.int8)
        elif style == 1:  # Sparse changes to a common fill.
            a = np.full(shape, colors[0], dtype=np.int8)
            mask = rng.random(shape) < rng.uniform(.25, .7)
            a[mask] = rng.choice(colors, np.count_nonzero(mask))
        elif style == 2:  # Intersecting row/column color patterns plus accents.
            n = len(colors)
            indices = (rng.integers(n, size=(h, 1)) + rng.integers(n, size=(1, w))) % n
            a = np.array(colors, dtype=np.int8)[indices]
            for _ in range(int(rng.integers(1, 4))):
                a[int(rng.integers(h)), int(rng.integers(w))] = int(rng.choice(colors))
        else:  # Repeated full vectors are legal when separated, never deduped.
            a = rng.choice(colors, shape).astype(np.int8)
            b = a if h >= 3 and (w < 3 or rng.random() < .5) else a.T
            if len(b) >= 3:
                i = int(rng.integers(len(b) - 2))
                j = int(rng.integers(i + 2, len(b)))
                b[j] = b[i]
        if rng.random() < .45:
            # A whole blank logical edge survives as one output band.
            _side(a, int(rng.integers(4)))[:] = palette["bg"]
        elif rng.random() < .25:
            a[rng.random(shape) < .12] = palette["bg"]
        if _visible(a) and _asymmetric(a):
            return a
    raise RuntimeError("bounded logical-motif sampling exhausted")


def _axis(rng, n, mode):
    if mode == "uniform":
        return np.full(n, int(rng.integers(2, min(5, 24 // n) + 1)), dtype=int)
    for _ in range(64):
        bands = rng.integers(1, 5, size=n)
        if mode == "strong":
            bands[int(rng.integers(n))] = int(rng.integers(6, 10))
        if bands.sum() <= 24 and max(bands) > 1 and len(set(bands)) > 1:
            return bands
    # Deterministic bounded fallback still meets each requested profile.
    bands = np.ones(n, dtype=int)
    bands[int(rng.integers(n))] = 6 if mode == "strong" else 2
    return bands


def _scales(rng, shape, teaching=False, strong=False, mild_only=False):
    if teaching:
        modes = ("mild", "mild")
    elif strong:
        modes = ("strong", "strong")
    else:
        mode = int(rng.integers(5))
        modes = (("uniform", "uniform"), ("uniform", "mild"),
                 ("mild", "uniform"), ("mild", "mild"),
                 ("mild" if mild_only else "strong", "mild"))[mode]
    return tuple(_axis(rng, n, m) for n, m in zip(shape, modes))


def _expanded(logical, rows, cols):
    return np.repeat(np.repeat(logical, rows, axis=0), cols, axis=1)


def _render(rng, logical, rows, cols, palette, inner, code):
    payload = _expanded(logical, rows, cols)
    h, w = payload.shape
    guides = {inner: palette["markers"][0], (inner + 1) % 4: palette["markers"][1]}
    controls = {_side_map(code, side): color for side, color in guides.items()}
    H = int(rng.integers(h + 6, min(30, h + 12) + 1))
    W = int(rng.integers(w + 6, min(30, w + 12) + 1))
    if rng.random() < .1:
        H = W = 30
    r = int(rng.integers(2 + (0 in guides), H - h - 1 - (2 in guides)))
    c = int(rng.integers(2 + (3 in guides), W - w - 1 - (1 in guides)))
    a = np.full((H, W), palette["bg"], dtype=np.int8)
    a[r:r + h, c:c + w] = payload
    for side, color in guides.items():
        _paint_guide(a, (r, c, h, w), side, color)
    for side, color in controls.items():
        _side(a, side)[1:-1] = color
    # Optional sparse border decoration has no marker or content identity.
    if rng.random() < .55:
        for side in set(range(4)) - set(controls):
            core = _side(a, side)[1:-1]
            marks = rng.random(len(core)) < rng.uniform(.1, .55)
            for i in range(2, len(marks)):
                if marks[i - 2] and marks[i - 1]:
                    marks[i] = False
            core[marks] = palette["noise"]
    return {"input": a, "output": _transform(logical, code)}


def _other_correspondence(inner, code):
    sides = (inner, (inner + 1) % 4)
    targets = tuple(_side_map(code, side) for side in sides)[::-1]
    choices = [k for k in range(8) if tuple(_side_map(k, s) for s in sides) == targets]
    assert len(choices) == 1
    return choices[0]


def _permute_profile(rng, bands):
    for _ in range(16):
        changed = rng.permutation(bands)
        if not np.array_equal(changed, bands):
            return changed
    # Nonuniform teaching profiles always have a nontrivial cyclic shift.
    for shift in range(1, len(bands)):
        changed = np.roll(bands, shift)
        if not np.array_equal(changed, bands):
            return changed
    raise AssertionError("teaching profile must be nonuniform")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    # Regimes change sampling, never the transformation or interpretation.
    regime = int(rng.integers(3))  # novel outer corner / stronger bands / extra logical band
    held = int(rng.integers(4))
    allowed = [(i, k) for i in range(4) for k in (1, 3)
               if regime != 0 or held not in {_side_map(k, i), _side_map(k, (i + 1) % 4)}]
    inner, first_code = allowed[int(rng.integers(len(allowed)))]
    palette = _palette(rng)
    logical = _teaching_logical(rng, palette, 5 if regime == 2 else 6)
    rows, cols = _scales(rng, logical.shape, teaching=True)
    first = _render(rng, logical, rows, cols, palette, inner, first_code)
    info = _decode(first["input"])
    second = {"input": first["input"].copy(), "output": None}
    r, c, h, w = info["bounds"]
    rows2, cols2 = _permute_profile(rng, rows), _permute_profile(rng, cols)
    second["input"][r:r + h, c:c + w] = _expanded(logical, rows2, cols2)
    replacement = dict(zip(info["controls"].values(), list(info["controls"].values())[::-1]))
    for side, color in info["controls"].items():
        _side(second["input"], side)[1:-1] = replacement[color]
    second["output"] = _transform(logical, _other_correspondence(inner, first_code))
    teaching = [first, second]
    if rng.random() < .5:
        teaching.reverse()
    puzzle = {"train": teaching, "test": []}
    used = {_key(p["input"]) for p in teaching}
    for split, count in (("train", nt - 2), ("test", ne)):
        for _ in range(count):
            for _attempt in range(32):
                p = _palette(rng)
                if regime == 2 and split == "test":
                    shape = (6, int(rng.integers(2, 6)))
                    if rng.random() < .5:
                        shape = shape[::-1]
                else:
                    cap = 5 if regime == 2 else 6
                    shape = tuple(map(int, rng.integers(2, cap + 1, size=2)))
                a = _logical(rng, p, shape)
                scales = _scales(rng, shape, strong=regime == 1 and split == "test",
                                 mild_only=regime == 1 and split == "train")
                arrangements = [(i, k) for i in range(4) for k in range(8)
                                if regime != 0 or
                                ((held in {_side_map(k, i), _side_map(k, (i + 1) % 4)}) == (split == "test"))]
                i, k = arrangements[int(rng.integers(len(arrangements)))]
                pair = _render(rng, a, *scales, p, i, k)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check local semantics, evidence for both parents, and a visible shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    information, used = {}, set()
    for split in ("train", "test"):
        information[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in used, "duplicate inputs"
            used.add(key)
            info = _decode(inp)
            a, rows, cols = info["logical"], info["rows"], info["cols"]
            assert all(2 <= n <= 6 for n in a.shape) and _visible(a)
            assert _asymmetric(a), "generated mosaics must make orientation observable"
            assert max(rows) > 1 and max(cols) > 1, "both-axis compression must matter"
            assert max(rows) <= 9 and max(cols) <= 9
            assert np.array_equal(_expanded(a, rows, cols), info["payload"])
            expected = _coordinate_orient(a, info["guides"], info["controls"])
            assert np.array_equal(out, expected), "incorrect logical cells, crop or orientation"
            assert np.array_equal(np.sort(out.ravel()), np.sort(a.ravel()))
            assert _key(out) != _key(_coordinate_orient(info["payload"], info["guides"], info["controls"]))
            if info["code"] != 0:
                assert _key(out) != _key(a), "orientation must affect asymmetric content"
            information[split].append(info)
    train, test = information["train"], information["test"]
    assert any(_single_seam(i["logical"]) and _separated_repeat(i["logical"])
               and _blank_sides(i["logical"], i["background"])
               and len(set(i["rows"])) > 1 and len(set(i["cols"])) > 1 for i in train)
    contrasts = [(a, b) for j, a in enumerate(train) for b in train[j + 1:]
                 if a["guides"] == b["guides"] and set(a["controls"]) == set(b["controls"])
                 and a["controls"] != b["controls"] and a["bounds"] == b["bounds"]
                 and np.array_equal(a["logical"], b["logical"])
                 and not np.array_equal(a["rows"], b["rows"])
                 and not np.array_equal(a["cols"], b["cols"])]
    assert contrasts, "training needs a correspondence/scale contrast"
    assert any(i["code"] in (1, 3) for i in train)
    assert any(i["code"] >= 4 for i in train)
    # At least one concrete shift must be visible without generator metadata.
    corners_train = {frozenset(i["controls"]) for i in train}
    new_corner = all(frozenset(i["controls"]) not in corners_train for i in test)
    largest_train = max(max(max(i["rows"]), max(i["cols"])) for i in train)
    larger_bands = all(min(max(i["rows"]), max(i["cols"])) > largest_train for i in test)
    largest_dim = max(max(i["logical"].shape) for i in train)
    larger_logical = all(max(i["logical"].shape) > largest_dim for i in test)
    assert new_corner or larger_bands or larger_logical, "missing meaningful train/test shift"
    return None


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded in-memory tests over 42 seeds, all counts, and targeted defects."""
    names = {s: set() for s in ("train", "test")}
    canvas, logical_shapes, profiles, palettes, structures, signatures = set(), set(), set(), set(), set(), set()
    blanks, repeats, seams, shift_types = set(), 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256(repr(_puzzle_key(puzzle)).encode()).digest()
        signatures.add(signature)
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        info_by_split = {}
        for split in ("train", "test"):
            info_by_split[split] = []
            for pair in puzzle[split]:
                i = _decode(pair["input"])
                info_by_split[split].append(i)
                a = i["logical"]
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                names[split].add(i["code"])
                canvas.add(pair["input"].shape)
                logical_shapes.add(a.shape)
                profiles.add((tuple(i["rows"]), tuple(i["cols"])))
                palettes.add((i["background"], tuple(sorted(i["controls"].values())), tuple(np.unique(a))))
                mapping = {}
                normalized = bytes(mapping.setdefault(int(v), len(mapping)) for v in a.ravel())
                structures.add((a.shape, normalized))
                blanks.update(_blank_sides(a, i["background"]))
                repeats += _separated_repeat(a)
                seams += _single_seam(a)
                # Swapping the two outer identities, with every interior pixel
                # unchanged, must change the answer for every asymmetric mosaic.
                changed = pair["input"].copy()
                colors = list(i["controls"].values())
                swap = dict(zip(colors, colors[::-1]))
                for side, color in i["controls"].items():
                    _side(changed, side)[1:-1] = swap[color]
                assert not np.array_equal(solve(changed), pair["output"])
        train, test = info_by_split["train"], info_by_split["test"]
        if all(frozenset(t["controls"]) not in {frozenset(a["controls"]) for a in train} for t in test):
            shift_types.add("corner")
        if max(max(a["logical"].shape) for a in train) < min(max(t["logical"].shape) for t in test):
            shift_types.add("logical")
        if max(max(max(a["rows"]), max(a["cols"])) for a in train) < min(min(max(t["rows"]), max(t["cols"])) for t in test):
            shift_types.add("bands")
    assert len(signatures) == 42
    assert names["train"] == names["test"] == set(range(8))
    assert len(canvas) >= 60 and len(logical_shapes) >= 15
    assert len(profiles) >= 100 and len(palettes) >= 80 and len(structures) >= 100
    assert blanks == set(range(4)) and repeats >= 84 and seams >= 84
    assert shift_types == {"corner", "logical", "bands"}
    # Independent coordinate transformation check on a non-square labeled grid.
    a = np.arange(15, dtype=np.int8).reshape(3, 5)
    for k in range(8):
        guides = {0: 8, 1: 9}
        controls = {_side_map(k, s): c for s, c in guides.items()}
        assert np.array_equal(_coordinate_orient(a, guides, controls), _transform(a, k))
    # Maximal runs retain separated duplicates, including identical complete
    # logical rows/columns separated by other vectors.
    a = np.array(((1, 1, 2, 1), (3, 4, 4, 3), (1, 1, 2, 1)), dtype=np.int8)
    expanded = _expanded(a, (2, 1, 3), (1, 3, 2, 1))
    assert np.array_equal(_compress(expanded)[0], a)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for defect in ("cell", "crop", "guide", "uncollapsed"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = bad["test"][0]
        i = _decode(pair["input"])
        if defect == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif defect == "crop":
            pair["output"] = pair["output"][1:]
        elif defect == "uncollapsed":
            pair["output"] = _coordinate_orient(i["payload"], i["guides"], i["controls"])
        else:
            side = next(iter(i["guides"]))
            _paint_guide(pair["input"], i["bounds"], side, i["background"])
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("defect accepted: " + defect)
    return {"seeds": 42, "pairs": 210, "symmetries_train": len(names["train"]),
            "symmetries_test": len(names["test"]), "canvas_shapes": len(canvas),
            "logical_shapes": len(logical_shapes), "scale_profiles": len(profiles),
            "logical_structures": len(structures), "palette_roles": len(palettes),
            "separated_repeat_pairs": int(repeats), "single_seam_pairs": int(seams),
            "shift_types": sorted(shift_types)}
