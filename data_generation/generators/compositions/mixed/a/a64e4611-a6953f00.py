"""Empty-window union, sampled at the corner addressed by square-side parity.

Background is 0; the newly introduced mark is 3. All other colors are obstacles.
No parent generator, files, seed, or demonstrations are consulted by solve().
"""
from __future__ import annotations

import hashlib
import numpy as np


BACKGROUND = 0
MARK = 3
OBSTACLES = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=np.int8)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _empty_centers(a):
    """Every window is centered in-grid; off-grid cells impose no condition."""
    h, w = a.shape
    padded = np.pad(a == BACKGROUND, 1, constant_values=True)
    centers = np.ones((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            centers &= padded[dy:dy + h, dx:dx + w]
    return centers


def _union_mask(a):
    centers = _empty_centers(a)
    h, w = a.shape
    padded = np.pad(centers, 1, constant_values=False)
    union = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            union |= padded[dy:dy + h, dx:dx + w]
    return union


def _marked(a):
    result = a.copy()
    result[_union_mask(a)] = MARK
    return result


def _crop(a, left):
    return a[:2, :2].copy() if left else a[:2, -2:].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Mark the union of original empty windows, then copy the parity corner."""
    a = _grid(input_grid)
    if a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("input must be square with side 2..30")
    if np.any(a == MARK):
        raise ValueError("3 is reserved for newly introduced marks")
    return _crop(_marked(a), bool(a.shape[0] % 2))


def _reference(a, *, clipped=True, centers_only=False):
    """Literal loop oracle for validation, separately from the vectorized solver."""
    n = a.shape[0]
    mask = np.zeros_like(a, dtype=bool)
    for r in range(n):
        for c in range(n):
            if not clipped and (r == 0 or c == 0 or r == n - 1 or c == n - 1):
                continue
            r0, r1 = max(0, r - 1), min(n, r + 2)
            c0, c1 = max(0, c - 1), min(n, c + 2)
            if np.all(a[r0:r1, c0:c1] == BACKGROUND):
                if centers_only:
                    mask[r, c] = True
                else:
                    mask[r0:r1, c0:c1] = True
    out = a.copy()
    out[mask] = MARK
    return out


def _corner_union(patch, clipped=True):
    # Only centers in the upper-left 3x3 can affect the upper-left 2x2.
    # Their neighborhoods lie entirely in this 4x4 context.
    mask = np.zeros((2, 2), dtype=bool)
    for r in range(3):
        for c in range(3):
            if not clipped and (r == 0 or c == 0):
                continue
            r0, c0 = max(0, r - 1), max(0, c - 1)
            if not patch[r0:r + 2, c0:c + 2].any():
                mask[r0:min(2, r + 2), c0:min(2, c + 2)] = True
    return mask


_TARGETS = {
    "one": np.array([[0, 0], [0, 1]], dtype=bool),
    "three": np.array([[0, 1], [1, 1]], dtype=bool),
    "row": np.array([[0, 0], [1, 1]], dtype=bool),
    "column": np.array([[0, 1], [0, 1]], dtype=bool),
    "full": np.ones((2, 2), dtype=bool),
    "none": np.zeros((2, 2), dtype=bool),
}


def _context(rng, kind):
    """A varied binary 4x4 context with a controlled union in its corner.

    One puts its obstacle at the inner top cell; three puts it at the outer
    top cell. Mirroring the right context makes their actual raw crops equal.
    Cells outside those crops distinguish the two empty-window answers.
    """
    if kind == "random":
        return rng.random((4, 4)) < float(rng.uniform(.08, .8))
    patch = np.zeros((4, 4), dtype=bool)
    if kind == "full":
        patch[:] = True
        patch[:2, :2] = False
    elif kind == "none":
        patch[:] = True
        patch[:2, :2] = [[1, 0], [0, 0]]
    else:
        patch[0, 1 if kind == "one" else 0] = True
        if kind == "row":
            patch[0, 2] = True
        if kind in ("one", "column"):
            patch[2, 0] = True
        if kind == "three":
            # Both outer arms need clipped windows; the diagonal blocks
            # the corresponding full 3x3 windows.
            patch[2, 2] = True
    target = _TARGETS[kind]
    assert np.array_equal(_corner_union(patch), target)
    for _ in range(24):
        r, c = (int(x) for x in rng.integers(0, 4, size=2))
        if r < 2 and c < 2:
            continue
        patch[r, c] = ~patch[r, c]
        keep = np.array_equal(_corner_union(patch), target)
        if kind == "three":
            keep &= not np.array_equal(_corner_union(patch, clipped=False), target)
        if not keep:
            patch[r, c] = ~patch[r, c]
    return patch


def _texture(rng, n):
    """Obstacle texture and negative-space geometry, independent of parity."""
    mode = int(rng.integers(0, 6))
    mask = rng.random((n, n)) < float(rng.uniform(.12, .62))
    if mode == 1:  # rooms in a dense field
        mask = rng.random((n, n)) < float(rng.uniform(.5, .9))
        for _ in range(int(rng.integers(2, 7))):
            h, w = (int(x) for x in rng.integers(2, max(3, n // 2 + 1), size=2))
            r, c = int(rng.integers(n - h + 1)), int(rng.integers(n - w + 1))
            mask[r:r + h, c:c + w] = False
    elif mode == 2:  # islands and streaks on an open field
        mask[:] = False
        for _ in range(int(rng.integers(2, n + 2))):
            r, c = (int(x) for x in rng.integers(n, size=2))
            h, w = (int(x) for x in rng.integers(1, max(2, n // 3 + 1), size=2))
            mask[r:min(n, r + h), c:min(n, c + w)] = True
    elif mode == 3:  # a stepped empty corridor, either orientation
        vertical = bool(rng.integers(2))
        center = int(rng.integers(n))
        for t in range(n):
            center = int(np.clip(center + int(rng.choice([-1, 0, 0, 1])), 0, n - 1))
            width = int(rng.integers(2, max(3, n // 3 + 1)))
            lo, hi = max(0, center - width // 2), min(n, center + (width + 1) // 2)
            if vertical:
                mask[t, lo:hi] = False
            else:
                mask[lo:hi, t] = False
    elif mode == 4:  # unequal bands, sometimes pierced by an island
        vertical = bool(rng.integers(2))
        for start in range(0, n, int(rng.integers(2, 6))):
            width = int(rng.integers(1, 5))
            if vertical:
                mask[:, start:min(n, start + width)] = False
            else:
                mask[start:min(n, start + width), :] = False
        if n >= 5:
            mask[n // 2, n // 2] = True
    elif mode == 5:  # dense field with scattered small pockets and thin slits
        mask[:] = True
        for _ in range(int(rng.integers(3, n + 3))):
            r, c = (int(x) for x in rng.integers(n, size=2))
            h, w = (int(x) for x in rng.integers(1, 6, size=2))
            mask[r:min(n, r + h), c:min(n, c + w)] = False
    return mask


def _paint_context(rng, a, left, kind, palette, corner_color):
    mask = _context(rng, kind)
    patch = np.zeros((4, 4), dtype=np.int8)
    patch[mask] = rng.choice(palette, size=int(mask.sum()))
    patch[:2, :2][mask[:2, :2]] = corner_color
    if left:
        a[:4, :4] = patch
    else:
        a[:4, -4:] = patch[:, ::-1]


def _new_input(rng, n, selected_kind, decoy_kind):
    palette = rng.choice(OBSTACLES, size=int(rng.integers(1, 9)), replace=False)
    occupied = _texture(rng, n)
    a = np.zeros((n, n), dtype=np.int8)
    a[occupied] = rng.choice(palette, size=int(occupied.sum()))
    if n >= 8:
        corner_color = int(rng.choice(palette))
        left = bool(n % 2)
        _paint_context(rng, a, left, selected_kind, palette, corner_color)
        _paint_context(rng, a, not left, decoy_kind, palette, corner_color)
    # Small test/extra grids use their natural overlapping contexts.
    return a


def _side(rng, parity, low, high, exclude=()):
    choices = [n for n in range(low, high + 1) if n % 2 == parity and n not in exclude]
    return int(rng.choice(choices))


def _threshold_separates(sides):
    """Whether a single side-length threshold could replace parity."""
    even = [n for n in sides if n % 2 == 0]
    odd = [n for n in sides if n % 2 == 1]
    return max(even) < min(odd) or max(odd) < min(even)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first_parity = int(rng.integers(2))
    parities = [first_parity, 1 - first_parity]
    train_sides = [_side(rng, p, 8, 18) for p in parities]
    if num_train >= 4:
        # Two sizes per parity alone can still make "small -> left, large ->
        # right" fit. Choose from the bounded set of interleaving size plans.
        options = [(a, b) for a in range(8, 19) for b in range(8, 19)
                   if a % 2 == parities[0] and b % 2 == parities[1]
                   and a not in train_sides and b not in train_sides
                   and not _threshold_separates(train_sides + [a, b])]
        train_sides.extend(options[int(rng.integers(len(options)))])
    while len(train_sides) < num_train:
        train_sides.append(int(rng.integers(8, 19)))
    if rng.integers(2):
        witness_kinds = ["one", "three"]
    else:
        witness_kinds = ["three", "one"]
    kinds = witness_kinds + ["full", "none"]
    choices = ["one", "three", "row", "column", "full", "none", "random"]
    puzzle = {"train": [], "test": []}
    seen = set()
    test_parity = int(rng.integers(2))
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                n = train_sides[i]
                kind = kinds[i] if i < 4 else str(rng.choice(choices))
                decoy = {"one": "three", "three": "one", "full": "none",
                         "none": "full"}.get(kind, str(rng.choice(choices)))
            else:
                parity = test_parity if i % 2 == 0 else 1 - test_parity
                # The first held-out example transfers to a genuinely new scale.
                # Further tests can overlap training sizes or reach side 2.
                n = _side(rng, parity, 21, 30) if i == 0 else _side(rng, parity, 2, 30)
                kind, decoy = (str(x) for x in rng.choice(choices, size=2))
            for _ in range(32):
                a = _new_input(rng, n, kind, decoy)
                key = _key(a)
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append({"input": a, "output": solve(a)})
                    break
            else:
                raise RuntimeError("unique input construction exhausted its bounded retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle):
    """Check semantics, schema, diagnostic witnesses, and the intended shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "split keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "pair counts")
    seen = set()
    witness = {0: False, 1: False}
    union_difference = boundary_difference = zero_difference = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            n = a.shape[0]
            _require(a.shape == (n, n) and n >= 2, "square input")
            _require(not np.any(a == MARK), "reserved mark in input")
            _require(out.shape == (2, 2), "output shape")
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            marked = _reference(a)
            left = bool(n % 2)
            expected = _crop(marked, left)
            _require(np.array_equal(out, expected), "empty-window union / parity rule")
            _require(np.array_equal(out, solve(a)), "solver agreement")
            union = marked == MARK
            _require(not np.any(union & (a != BACKGROUND)), "marked an obstacle")
            _require(np.all(marked[a != BACKGROUND] == a[a != BACKGROUND]), "obstacle changed")
            _require(np.all(union[_empty_centers(a)]), "empty centers missing from union")
            if split == "train":
                decoy = _crop(marked, not left)
                has_marks = bool(np.any(out == MARK))
                witness[n % 2] |= has_marks and not np.array_equal(out, decoy)
                union_difference |= not np.array_equal(out, _crop(_reference(a, centers_only=True), left))
                boundary_difference |= not np.array_equal(out, _crop(_reference(a, clipped=False), left))
                all_zeros = _crop(a, left)
                all_zeros[all_zeros == BACKGROUND] = MARK
                zero_difference |= not np.array_equal(out, all_zeros)
    _require(all(witness.values()), "both parities need informative marked corners")
    _require(union_difference, "training fails to distinguish union from centers")
    _require(boundary_difference, "training fails to demonstrate clipping")
    _require(zero_difference, "training fails to reject marking every zero")
    # First two examples deliberately keep raw corners identical: local context,
    # rather than literal crop content, supplies the difference used by parity.
    for pair in puzzle["train"][:2]:
        a, out = np.asarray(pair["input"]), np.asarray(pair["output"])
        _require(np.array_equal(_crop(a, True), _crop(a, False)), "raw witness corners differ")
        _require(int(np.sum(out == MARK)) in (1, 3), "witness mark count")
        _require(not np.array_equal(out, _crop(a, bool(a.shape[0] % 2))), "A ablation invisible")
    _require({int(np.sum(np.asarray(p["output"]) == MARK))
              for p in puzzle["train"][:2]} == {1, 3}, "one/three witnesses")
    for pair, mark_count in zip(puzzle["train"][2:4], (4, 0)):
        a, out = np.asarray(pair["input"]), np.asarray(pair["output"])
        _require(int(np.sum(out == MARK)) == mark_count, "full/uncovered demonstration")
        decoy = _crop(_reference(a), not bool(a.shape[0] % 2))
        _require(not np.array_equal(out, decoy), "additional parity witness is ambiguous")
    if tr >= 4:
        train_sides = [np.asarray(p["input"]).shape[0] for p in puzzle["train"]]
        for parity in (0, 1):
            sizes = {n for n in train_sides if n % 2 == parity}
            _require(len(sizes) >= 2, "each parity needs two training scales")
        _require(not _threshold_separates(train_sides[:4]),
                 "training permits a side-length threshold instead of parity")
    _require(np.asarray(puzzle["test"][0]["input"]).shape[0]
             > max(np.asarray(p["input"]).shape[0] for p in puzzle["train"]), "scale shift")
    if te >= 2:
        _require({np.asarray(p["input"]).shape[0] % 2 for p in puzzle["test"]} == {0, 1},
                 "multiple tests should cover both parities")


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Stream 42 seeds; retain only bounded sets of hashes and small counters."""
    signatures, structures, sizes, palettes, outputs, test_parities = (set() for _ in range(6))
    nonempty = empty = clipping = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256(repr(_signature(puzzle)).encode()).digest()
        signatures.add(digest)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "determinism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                sizes.add(a.shape[0])
                palettes.add(tuple(int(c) for c in np.unique(a)))
                structures.add((a.shape, hashlib.sha256((a != 0).tobytes()).digest()))
                outputs.add((a.shape[0] % 2, tuple((out == MARK).flat)))
                nonempty += bool(np.any(out == MARK))
                empty += not np.any(out == MARK)
                clipping += not np.array_equal(out, _crop(_reference(a, clipped=False), bool(a.shape[0] % 2)))
                if split == "test":
                    test_parities.add(a.shape[0] % 2)
        # Color relabeling cannot change emptiness; obstacles must still copy.
        a = puzzle["test"][0]["input"]
        mapping = np.arange(10, dtype=np.int8)
        mapping[OBSTACLES] = OBSTACLES[::-1]
        _require(np.array_equal(solve(mapping[a]), mapping[solve(a)]), "obstacle relabeling")
    _require(len(signatures) == 42, "distinct seed puzzles")
    _require(len(structures) >= 180 and len(sizes) >= 18, "structural / scale diversity")
    _require(len(palettes) >= 40 and test_parities == {0, 1}, "palette / test parity diversity")
    _require(len(outputs) >= 10 and nonempty and empty and clipping >= 42, "mask diversity")
    # Every legal explicit count; only one puzzle is live at a time.
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(10000 + 10 * tr + te, tr, te)
            _require((len(p["train"]), len(p["test"])) == (tr, te), "explicit counts")
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (None, 1), (4, None), (True, 1), (4.0, 1), (4, "1")):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {args}")
    # Exhaustive small grids cover clipping, no-window, and all-window extremes.
    for n in (2, 3):
        for bits in range(1 << (n * n)):
            a = np.array([(bits >> i) & 1 for i in range(n * n)], dtype=np.int8).reshape(n, n)
            expected = _crop(_reference(a), bool(n % 2))
            _require(np.array_equal(solve(a), expected), "exhaustive small-grid rule")
    # The solver's domain extends beyond generated witness templates. Test every
    # side, all-background/all-obstacle extremes, and uncontrolled random fields.
    rng = np.random.default_rng(6789)
    for n in range(2, 31):
        for density in (0., .2, .8, 1.):
            a = rng.choice(OBSTACLES, size=(n, n))
            a[rng.random((n, n)) >= density] = BACKGROUND
            expected = _crop(_reference(a), bool(n % 2))
            a.flags.writeable = False
            _require(np.array_equal(solve(a), expected), "uncontrolled input / immutable input")
            _require(np.array_equal(solve(a.tolist()), expected), "nested-list input")
            if density == .2 and n >= 8:
                # These positions lie outside both candidates' receptive fields.
                b = a.copy()
                b[4:, :] = rng.choice(np.append(OBSTACLES, BACKGROUND), size=b[4:, :].shape)
                b[:4, 4:-4] = BACKGROUND
                _require(np.array_equal(solve(a), solve(b)), "distant clutter affected the crop")
    # A 2x2 corner pocket qualifies even when every surrounding cell is occupied.
    a = np.ones((9, 9), dtype=np.int8)
    a[:2, :2] = 0
    _require(np.array_equal(solve(a), np.full((2, 2), MARK)), "clipped corner window")
    # Re-using the same two corner contexts at opposite parities changes output.
    rng = np.random.default_rng(456)
    answers = []
    for n in (8, 9):
        a = np.ones((n, n), dtype=np.int8)
        _paint_context(rng, a, True, "one", np.array([1]), 1)
        _paint_context(rng, a, False, "three", np.array([1]), 1)
        answers.append(int(np.sum(solve(a) == MARK)))
    _require(answers == [3, 1], "parity must causally select the marked context")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, exhaustive sides 2 and 3)")
