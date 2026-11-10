"""Normalize repeated physical vectors, then decode a sparse rectangle code.

No file access, generation state, learned conventions, or parent imports are
needed by solve. All random choices use the supplied default_rng stream.
"""
from __future__ import annotations

import hashlib
import numpy as np

AXES = (1, 2, 3, 4, 6, 8)
CANVAS = 26
GAP = 2
PERIOD = 24


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0) or np.any(array > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return array.astype(np.int8, copy=False)


def _axis_indices(grid, axis):
    vectors = grid if axis == 0 else grid.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _collapse(grid):
    # Both masks refer to full ORIGINAL vectors. Dropping exact duplicate rows
    # cannot turn unequal columns equal (or vice versa), so orders commute.
    return grid[np.ix_(_axis_indices(grid, 0), _axis_indices(grid, 1))].copy()


def _axis_lengths(grid, axis):
    starts = np.flatnonzero(_axis_indices(grid, axis))
    return tuple(int(n) for n in np.diff(np.r_[starts, grid.shape[axis]]))


def _lattice(coarse, background):
    rows, cols = coarse.shape
    result = np.full((2 * rows + 1, 2 * cols + 1), background, dtype=np.int8)
    result[1::2, 1::2] = coarse
    return result


def _decode(input_grid):
    source = _grid(input_grid)
    canonical = _collapse(source)
    h, w = canonical.shape
    background = int(source[0, 0])
    if (h < 3 or w < 3 or h % 2 != 1 or w % 2 != 1
            or not np.all(canonical[::2] == background)
            or not np.all(canonical[:, ::2] == background)):
        raise ValueError("normalized input must have a complete alternating background scaffold")
    coarse = canonical[1::2, 1::2].copy()
    if (coarse.shape[0] not in AXES or coarse.shape[1] not in AXES
            or not np.all(np.any(coarse != background, axis=0))
            or not np.all(np.any(coarse != background, axis=1))):
        raise ValueError("unsupported or unobservable coarse axis")
    return coarse, background, canonical


def _runs(row, background):
    """Triples (color, left-inclusive, right-exclusive)."""
    result = []
    c = 0
    while c < len(row):
        color = int(row[c])
        right = c + 1
        while right < len(row) and int(row[right]) == color:
            right += 1
        if color != background:
            result.append((color, c, right))
        c = right
    return result


def _render_stacks(coarse, background):
    rows, cols = coarse.shape
    py, px = PERIOD // rows, PERIOD // cols
    out = np.full((CANVAS, CANVAS), background, dtype=np.int8)
    runs = [set(_runs(row, background)) for row in coarse]
    for r, row in enumerate(runs):
        for color, left, right in sorted(row):
            key = (color, left, right)
            if r and key in runs[r - 1]:
                continue
            end = r + 1
            while end < rows and key in runs[end]:
                end += 1
            out[GAP + r * py:end * py,
                GAP + left * px:right * px] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution, including canonical (unstretched) family members."""
    coarse, background, _ = _decode(input_grid)
    return _render_stacks(coarse, background)


def _render_bridges(coarse, background, vertical="exact"):
    """Generation/check path: paint unit cores and local connecting bridges.

    vertical='none' and 'overlap' are deliberately wrong necessity ablations.
    """
    rows, cols = coarse.shape
    py, px = PERIOD // rows, PERIOD // cols
    out = np.full((CANVAS, CANVAS), background, dtype=np.int8)
    runs = [set(_runs(row, background)) for row in coarse]
    for r, row in enumerate(coarse):
        for c, value in enumerate(row):
            color = int(value)
            if color == background:
                continue
            y, x = GAP + r * py, GAP + c * px
            out[y:(r + 1) * py, x:(c + 1) * px] = color
            if c + 1 < cols and int(row[c + 1]) == color:
                out[y:(r + 1) * py, (c + 1) * px:(c + 1) * px + GAP] = color
        if r + 1 == rows or vertical == "none":
            continue
        for color, left, right in sorted(runs[r] & runs[r + 1]):
            out[(r + 1) * py:(r + 1) * py + GAP,
                GAP + left * px:right * px] = color
        if vertical == "overlap":
            for c in range(cols):
                color = int(coarse[r, c])
                if color != background and int(coarse[r + 1, c]) == color:
                    out[(r + 1) * py:(r + 1) * py + GAP,
                        GAP + c * px:(c + 1) * px] = color
    return out


def _features(coarse, background):
    runs = [set(_runs(row, background)) for row in coarse]
    horizontal = any(right - left >= 2 for row in runs for _, left, right in row)
    exact = any(a & b for a, b in zip(runs, runs[1:]))
    unequal = any(ca == cb and (la, ra) != (lb, rb) and max(la, lb) < min(ra, rb)
                  for a, b in zip(runs, runs[1:])
                  for ca, la, ra in a for cb, lb, rb in b)
    repeated_rows = bool(np.any(np.all(coarse[1:] == coarse[:-1], axis=1)))
    return horizontal, bool(exact), unequal, repeated_rows


def _paint(coarse, row, left, right, color, background):
    coarse[row, left:right] = color
    if left and int(coarse[row, left - 1]) == color:
        coarse[row, left - 1] = background
    if right < coarse.shape[1] and int(coarse[row, right]) == color:
        coarse[row, right] = background


def _fill_axes(coarse, background, colors, rng, protected_rows=()):
    rows, cols = coarse.shape
    for r in range(rows):
        if np.all(coarse[r] == background):
            coarse[r, int(rng.integers(cols))] = int(rng.choice(colors))
    available_rows = [r for r in range(rows) if r not in protected_rows]
    for c in range(cols):
        if np.all(coarse[:, c] == background):
            r = int(rng.choice(available_rows or list(range(rows))))
            coarse[r, c] = int(rng.choice(colors))


def _sample_coarse(rng, shape, background, role):
    rows, cols = shape
    k = int(rng.integers(2 if role == "unequal" else 1, min(7, rows * cols, 9) + 1))
    palette = np.array([c for c in range(10) if c != background], dtype=np.int8)
    colors = rng.choice(palette, size=k, replace=False)
    density = float(rng.choice((0.35, 0.5, 0.7, 0.9, 1.0)))
    coarse = rng.choice(colors, size=shape).astype(np.int8)
    coarse[rng.random(shape) > density] = background
    _fill_axes(coarse, background, colors, rng)
    # Local compositions (not fixed whole-grid templates) create long runs,
    # partial overlaps, isolated labels and later returns with independent noise.
    for _ in range(int(rng.integers(1, 5))):
        r = int(rng.integers(rows))
        left = int(rng.integers(cols))
        right = int(rng.integers(left + 1, cols + 1))
        color = int(rng.choice(colors))
        _paint(coarse, r, left, right, color, background)
        if r + 1 < rows and rng.random() < 0.45:
            depth = int(rng.integers(2, rows - r + 1))
            for lower in range(r + 1, r + depth):
                _paint(coarse, lower, left, right, color, background)
    protected = ()
    if role == "exact":
        top = int(rng.integers(rows - 1))
        left = int(rng.integers(cols - 1))
        right = int(rng.integers(left + 2, cols + 1))
        _paint(coarse, top, left, right, int(rng.choice(colors)), background)
        coarse[top + 1] = coarse[top]
        protected = (top, top + 1)
    elif role == "unequal":
        spans = [(a, b) for a in range(cols) for b in range(a + 1, cols + 1)]
        choices = [(a, b) for a in spans for b in spans
                   if a != b and max(a[0], b[0]) < min(a[1], b[1])]
        a, b = choices[int(rng.integers(len(choices)))]
        top = int(rng.integers(rows - 1))
        color = int(rng.choice(colors))
        _paint(coarse, top, *a, color, background)
        _paint(coarse, top + 1, *b, color, background)
    _fill_axes(coarse, background, colors, rng, protected)
    return coarse


def _stretch_lengths(rng, axis_count, mode):
    """Positive repeats with an exact 30-cell cap; never unbounded rejection."""
    size = 2 * axis_count + 1
    lengths = np.ones(size, dtype=int)
    if mode == "none":
        return lengths
    extra = int(rng.integers(1, 31 - size))
    # Space-only retains isolated dots. General modes also repeat signal vectors;
    # the repeated physical signal is NOT extra semantic rectangle height.
    if mode == "space":
        choices = np.arange(0, size, 2)
    elif mode == "signal":
        choices = np.arange(1, size, 2)
    else:
        choices = np.arange(size)
    if mode == "mixed":
        forced = int(rng.choice(np.arange(1, size, 2)))
        lengths[forced] += 1
        extra -= 1
    if extra:
        if rng.random() < 0.4:
            lengths[int(rng.choice(choices))] += extra
        else:
            lengths[choices] += rng.multinomial(extra, np.ones(len(choices)) / len(choices))
    return lengths


def _key(grid):
    return grid.shape, grid.tobytes()


def _digest_puzzle(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    held = int(rng.choice((3, 4, 6, 8)))
    familiar = [n for n in AXES if n != held]
    first_pool = [(r, c) for r in familiar if r >= 3 for c in familiar if c >= 3]
    second_pool = [(r, c) for r in familiar if r >= 2 for c in familiar if c >= 3]
    first = first_pool[int(rng.integers(len(first_pool)))]
    second_choices = [s for s in second_pool if s != first]
    second = second_choices[int(rng.integers(len(second_choices)))]
    train_specs = [(first, "exact", "space"), (second, "unequal", "mixed")]
    while len(train_specs) < nt:
        shape = (int(rng.choice(familiar)), int(rng.choice(familiar)))
        train_specs.append((shape, "random", "random"))
    train_specs = [train_specs[int(i)] for i in rng.permutation(nt)]
    test_pool = [(r, c) for r in AXES for c in AXES if r == held or c == held]
    test_specs = []
    for i in rng.choice(len(test_pool), size=ns, replace=False):
        shape = test_pool[int(i)]
        possible = ["random"]
        if shape[0] >= 3 and shape[1] >= 2:
            possible.append("exact")
        if shape[0] >= 2 and shape[1] >= 2:
            possible.append("unequal")
        test_specs.append((shape, str(rng.choice(possible)), "random"))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for shape, role, mode in specs:
            for _ in range(96):
                background = int(rng.integers(10))
                coarse = _sample_coarse(rng, shape, background, role)
                horizontal, exact, unequal, repeated = _features(coarse, background)
                if role == "exact" and not (horizontal and exact and repeated):
                    continue
                if role == "unequal" and not unequal:
                    continue
                if mode == "random":
                    ym, xm = (str(rng.choice(("space", "signal", "mixed"))) for _ in range(2))
                    if rng.random() < 0.22:
                        if rng.random() < 0.5:
                            ym = "none"
                        else:
                            xm = "none"
                else:
                    ym = xm = mode
                ys = _stretch_lengths(rng, shape[0], ym)
                xs = _stretch_lengths(rng, shape[1], xm)
                inp = np.repeat(np.repeat(_lattice(coarse, background), ys, axis=0), xs, axis=1)
                if _key(inp) in used:
                    continue
                used.add(_key(inp))
                puzzle[split].append({"input": inp, "output": _render_bridges(coarse, background)})
                break
            else:
                raise RuntimeError("bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle keys must be exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    if not (2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns):
        raise ValueError("illegal pair counts")
    seen, train_shapes = set(), set()
    features = [False] * 4
    two_axis = signal_repeat = nonuniform = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pair keys must be exactly input/output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            coarse, background, canonical = _decode(inp)
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            ys, xs = _axis_lengths(inp, 0), _axis_lengths(inp, 1)
            assert np.array_equal(np.repeat(np.repeat(canonical, ys, axis=0), xs, axis=1), inp)
            assert np.array_equal(_collapse(canonical), canonical), "collapse is not idempotent"
            assert max(ys) > 1 or max(xs) > 1, "generated pairs must use physical repetition"
            assert out.shape == (26, 26)
            assert np.array_equal(out, solve(inp))
            assert np.array_equal(out, _render_bridges(coarse, background))
            assert set(np.unique(inp)) == set(np.unique(out)), "colors must be copied"
            assert np.all(out[:GAP] == background) and np.all(out[-GAP:] == background)
            assert np.all(out[:, :GAP] == background) and np.all(out[:, -GAP:] == background)
            f = _features(coarse, background)
            if f[1]:
                assert not np.array_equal(out, _render_bridges(coarse, background, "none"))
            if f[2]:
                assert not np.array_equal(out, _render_bridges(coarse, background, "overlap"))
            if split == "train":
                train_shapes.add(coarse.shape)
                features = [a or b for a, b in zip(features, f)]
                two_axis |= max(ys) > 1 and max(xs) > 1
                signal_repeat |= max(ys[1::2]) > 1 or max(xs[1::2]) > 1
                nonuniform |= len(set(ys)) > 1 and len(set(xs)) > 1
    assert all(features), "training must expose runs, exact/unequal spans, and repeated coarse rows"
    assert two_axis and signal_repeat and nonuniform, "normalization evidence missing"
    assert len(train_shapes) >= 2
    trained_axes = {n for shape in train_shapes for n in shape}
    test_shapes = set()
    for pair in puzzle["test"]:
        coarse, _, _ = _decode(pair["input"])
        assert any(n not in trained_axes for n in coarse.shape), "test must have a held-out axis count"
        test_shapes.add(coarse.shape)
    assert len(test_shapes) == ns


def _normalized_signature(coarse, background):
    roles, labels = {background: 0}, []
    for value in coarse.flat:
        color = int(value)
        if color not in roles:
            roles[color] = len(roles)
        labels.append(roles[color])
    return coarse.shape, bytes(labels)


def self_test():
    """Bounded streaming checks: 42 seeds, all legal counts, and counterexamples.

    Only hashes/counters are retained. No samples, execution traces or logs are
    written. Returned compact-serializable metrics make the scan reviewable.
    """
    shapes, layouts, inputs, backgrounds, digests = set(), set(), set(), set(), set()
    test_shapes, palettes, rectangles = set(), set(), set()
    exact_count = unequal_count = row_repeat_count = no_op_count = 0
    pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _digest_puzzle(puzzle) == _digest_puzzle(generate_puzzle(seed))
        digests.add(_digest_puzzle(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                coarse, background, canonical = _decode(inp)
                shapes.add(coarse.shape)
                inputs.add(inp.shape)
                layouts.add(_normalized_signature(coarse, background))
                backgrounds.add(background)
                palettes.add(len(set(np.unique(coarse)) - {background}))
                f = _features(coarse, background)
                exact_count += f[1]
                unequal_count += f[2]
                row_repeat_count += f[3]
                no_op_count += (inp.shape[0] == canonical.shape[0]) != (inp.shape[1] == canonical.shape[1])
                runsets = [set(_runs(row, background)) for row in coarse]
                rectangles.add(sum(len(row - (runsets[r - 1] if r else set()))
                                   for r, row in enumerate(runsets)))
                if split == "test":
                    test_shapes.add(coarse.shape)
                assert np.array_equal(solve(canonical), out)
                # Expansion amounts, including signal thickness, are irrelevant.
                yy = np.ones(canonical.shape[0], dtype=int)
                xx = np.ones(canonical.shape[1], dtype=int)
                yy[0] += 30 - len(yy)
                xx[-1] += 30 - len(xx)
                stretched = np.repeat(np.repeat(canonical, yy, axis=0), xx, axis=1)
                assert np.array_equal(solve(stretched), out)
                # Independent simultaneous collapse equals either sequential order.
                a = inp[_axis_indices(inp, 0)]
                a = a[:, _axis_indices(a, 1)]
                b = inp[:, _axis_indices(inp, 1)]
                b = b[_axis_indices(b, 0)]
                assert np.array_equal(a, canonical) and np.array_equal(b, canonical)
                pairs_checked += 1
    assert len(digests) == 42 and len(backgrounds) == 10
    assert len(shapes) >= 20 and len(test_shapes) >= 12 and len(inputs) >= 60
    assert len(layouts) >= 130 and len(palettes) >= 5 and len(rectangles) >= 12
    assert exact_count >= 42 and unequal_count >= 42 and row_repeat_count >= 42
    assert no_op_count >= 8
    legal_calls = 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(18001 + nt * 4 + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
            legal_calls += 1
    invalid = ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
               (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None))
    for nt, ns in invalid:
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for bad_seed in (-1, 1.5, True):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # A legal 3x3 code becomes a plausible but WRONG 4x4 code without B.
    code = np.array([[2, 2, 3], [2, 2, 3], [4, 2, 2]], dtype=np.int8)
    canonical = _lattice(code, 0)
    repeats = np.ones(7, dtype=int)
    repeats[0] = 3
    source = np.repeat(np.repeat(canonical, repeats, axis=0), repeats, axis=1)
    correct = solve(source)
    assert not np.array_equal(correct, _render_stacks(source[1::2, 1::2], 0))
    for axis in (0, 1):
        reduced = source[_axis_indices(source, 0)] if axis == 0 else source[:, _axis_indices(source, 1)]
        assert not np.array_equal(correct, _render_stacks(reduced[1::2, 1::2], 0))
    assert not np.array_equal(correct, _render_bridges(code, 0, "none"))
    assert not np.array_equal(correct, _render_bridges(code, 0, "overlap"))
    # Collapsing AFTER extraction destroys the semantic repeated rows/columns.
    assert not np.array_equal(correct, _render_stacks(_collapse(code), 0))
    return {"seeds": 42, "pairs": pairs_checked, "legal_count_calls": legal_calls,
            "distinct_puzzles": len(digests), "coarse_shapes": len(shapes),
            "test_coarse_shapes": len(test_shapes), "input_shapes": len(inputs),
            "color_normalized_layouts": len(layouts), "background_colors": len(backgrounds),
            "payload_palette_sizes": sorted(palettes), "rectangle_counts": sorted(rectangles),
            "exact_merge_pairs": exact_count, "unequal_overlap_pairs": unequal_count,
            "repeated_coarse_row_pairs": row_repeat_count, "one_noop_axis_pairs": no_op_count,
            "necessity_ablation_fixture_passed": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
