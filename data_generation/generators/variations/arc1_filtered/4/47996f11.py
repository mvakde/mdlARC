"""Dihedral-symmetry restoration puzzles with varied designs, crops, and masks."""

import hashlib

import numpy as np


_SOURCE_INPUT_HASHES = frozenset({
    "01c596a86b8c8e3578b7fdd6df98a02cbc7cfe73a04f6a55ff718f4faa27f074",
    "e915b4e863abf7755d59d20fadcacaebc47fa35d7c1b88454dbd4708c079bbd6",
    "567b419809829825fef57386d885ea93b1a2018eb82b0d3a4080decb1c0451e2",
    "707dd42cf0853d772ebef14b7b147c2e6513c284f7e74c91f59cea998ac93f01",
    "3eb2740d1a8253dede59f34a088ee4e9160ca9e62d64b3d628b967db2335f090",
})


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("Illegal pair count")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("Training count must exceed test count")
    return int(num_train), int(num_test)


def _orbit(r, c, sy, sx, h, w):
    dy, dx = 2 * r - sy, 2 * c - sx
    offsets = (
        (dy, dx), (dy, -dx), (-dy, dx), (-dy, -dx),
        (dx, dy), (dx, -dy), (-dx, dy), (-dx, -dy),
    )
    return {
        ((sy + y) // 2, (sx + x) // 2)
        for y, x in offsets
        if (sy + y) % 2 == 0 and (sx + x) % 2 == 0
        and 0 <= (sy + y) // 2 < h and 0 <= (sx + x) // 2 < w
    }


def _design(rng, h, w, sy, sx, palette):
    # The unordered pair of distances identifies a full D4 orbit.
    tile = int(rng.integers(1, 6))
    period = int(rng.integers(3, 9))
    block = rng.integers(len(palette), size=(35, 35))
    local = rng.integers(len(palette), size=(period, period))
    rings = rng.integers(len(palette), size=35)
    spokes = rng.integers(len(palette), size=35)
    base = int(rng.integers(3))
    use_rings = rng.random() < 0.7
    use_spokes = rng.random() < 0.65
    use_motif = rng.random() < 0.55
    use_core = rng.random() < 0.45
    ring_width = int(rng.integers(2, 6))
    spoke_width = int(rng.integers(1, 4))
    core_radius = int(rng.integers(3, 9))
    core_color = int(rng.integers(len(palette)))
    out = np.empty((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            u, v = sorted((abs(2 * r - sy) // 2, abs(2 * c - sx) // 2))
            if base == 0:
                color = int(block[u // tile, v // tile])
            elif base == 1:
                color = int(rings[v // ring_width])
                if (u // tile + v // tile) % 3 == 0:
                    color = int(block[u // tile, v // tile])
            else:
                color = int(local[u % period, v % period])
                if v // tile % 3 == 0:
                    color = int(block[u // tile, v // tile])
            if use_rings and v % period < ring_width // 2 + 1:
                color = int(rings[v // ring_width])
            if use_spokes and (v - u) % period < spoke_width:
                color = int(spokes[(v - u) // period])
            if use_motif and u % period <= 1 and v % period <= 1:
                color = int(local[u % period, v % period])
            if use_core and v < core_radius:
                color = core_color
            out[r, c] = palette[color]
    return out


def _recoverable(mask, sy, sx):
    h, w = mask.shape
    for rr, cc in zip(*np.where(mask)):
        if not any(
            not mask[y, x]
            for y, x in _orbit(int(rr), int(cc), sy, sx, h, w)
        ):
            return False
    return True


def _indirect(mask, sy, sx):
    h, w = mask.shape
    for rr, cc in zip(*np.where(mask)):
        r, c = int(rr), int(cc)
        hr, vc = sy - r, sx - c
        if (
            0 <= hr < h and 0 <= vc < w
            and mask[hr, c] and mask[r, vc]
            and any(not mask[y, x] for y, x in _orbit(r, c, sy, sx, h, w))
        ):
            return True
    return False


def _patch(rng, mask, cy, cx, area, shape, margin=0):
    h, w = mask.shape
    aspect = float(rng.uniform(0.45, 2.2))
    ph = max(2, min(11, int(round((area * aspect) ** 0.5))))
    pw = max(2, min(11, int(round(area / ph))))
    top = max(margin, min(h - margin - ph, cy - int(rng.integers(ph))))
    left = max(margin, min(w - margin - pw, cx - int(rng.integers(pw))))
    if shape == 0:
        mask[top:top + ph, left:left + pw] = True
    elif shape == 1:
        arm_h = max(2, ph // 2)
        arm_w = max(2, pw // 2)
        mask[top:top + ph, left:left + arm_w] = True
        mask[top:top + arm_h, left:left + pw] = True
    else:
        split = max(1, ph // 2)
        mask[top:top + split, left:left + pw] = True
        mask[top + split:top + ph, left + min(2, pw - 2):left + pw] = True


def _random_anchor(rng, length):
    if rng.random() < 0.27:
        return int(rng.choice((0, 1, length - 2, length - 1)))
    return int(rng.integers(1, length - 1))


def _mask(rng, h, w, sy, sx, category, hard, interior_only):
    ranges = ((0.025, 0.075), (0.075, 0.155), (0.15, 0.27))
    low, high = ranges[category]
    target = float(rng.uniform(low, high))
    mask = np.zeros((h, w), dtype=bool)
    if hard:
        # Cover a cell and both direct mirrors, leaving rotated witnesses.
        for _ in range(30):
            r = int(rng.integers(3, h - 3))
            c = int(rng.integers(3, w - 3))
            hr, vc = sy - r, sx - c
            if (
                2 <= hr < h - 2 and 2 <= vc < w - 2
                and abs(hr - r) >= 5 and abs(vc - c) >= 5
                and len(_orbit(r, c, sy, sx, h, w)) >= 6
            ):
                for y, x in ((r, c), (hr, c), (r, vc)):
                    _patch(rng, mask, y, x, float(rng.uniform(8, 19)), 0,
                           int(interior_only))
                break
        else:
            return None
    count_ranges = ((1, 4), (2, 6), (3, 8))
    low_count, high_count = count_ranges[category]
    count = int(rng.integers(low_count, high_count))
    remaining = max(8.0, target * h * w - int(mask.sum()))
    for _ in range(count):
        area = float(remaining / count * rng.uniform(0.75, 1.8))
        cy = int(rng.integers(3, h - 3)) if interior_only else _random_anchor(rng, h)
        cx = int(rng.integers(3, w - 3)) if interior_only else _random_anchor(rng, w)
        _patch(rng, mask, cy, cx, area, int(rng.integers(3)),
               int(interior_only))
    fraction = float(mask.mean())
    if not (low * 0.65 <= fraction <= high + 0.035):
        return None
    if hard and not _indirect(mask, sy, sx):
        return None
    if not _recoverable(mask, sy, sx):
        return None
    return mask


def _compatible_centers(inp, mask_color):
    h, w = inp.shape
    masked = inp == mask_color
    yy, xx = np.indices((h, w))
    good = []
    for sy in range(h - 9, h + 8):
        for sx in range(w - 9, w + 8):
            if (sy - sx) % 2:
                continue
            dy, dx = 2 * yy - sy, 2 * xx - sx
            valid_center = True
            comparisons = 0
            for a, b in (
                (dy, -dx), (-dy, dx), (-dy, -dx),
                (dx, dy), (dx, -dy), (-dx, dy), (-dx, -dy),
            ):
                ty = (sy + a) // 2
                tx = (sx + b) // 2
                valid = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
                y = np.clip(ty, 0, h - 1)
                x = np.clip(tx, 0, w - 1)
                known = valid & ~masked & ~masked[y, x]
                comparisons += int(np.count_nonzero(known))
                if np.any(known & (inp != inp[y, x])):
                    valid_center = False
                    break
            if valid_center and comparisons >= 20:
                good.append((sy, sx))
    return good


def _restore(inp, mask_color, sy, sx):
    h, w = inp.shape
    mask = inp == mask_color
    out = inp.copy()
    for rr, cc in zip(*np.where(mask)):
        r, c = int(rr), int(cc)
        values = {
            int(inp[y, x])
            for y, x in _orbit(r, c, sy, sx, h, w)
            if not mask[y, x]
        }
        assert len(values) == 1
        out[r, c] = values.pop()
    return out


def _pair(rng, mask_color, category, hard):
    interior_only = bool(rng.random() < 0.3)
    for _ in range(6):
        h = int(rng.integers(19, 31))
        w = int(rng.integers(19, 31))
        if (h - w) % 2:
            w = w - 1 if w == 30 else w + 1
        sy = h - 1 + 2 * int(rng.integers(-2, 3))
        sx = w - 1 + 2 * int(rng.integers(-2, 3))
        allowed = np.array([c for c in range(10) if c != mask_color], dtype=np.uint8)
        k = int(rng.integers(4, 10))
        palette = rng.choice(allowed, size=k, replace=False)
        out = _design(rng, h, w, sy, sx, palette)
        if len(np.unique(out)) < 4:
            continue
        for _ in range(60):
            mask = _mask(rng, h, w, sy, sx, category, hard, interior_only)
            if mask is None:
                continue
            if category == 0 and not hard and (
                float(mask.mean()) >= 0.085 or _indirect(mask, sy, sx)
            ):
                continue
            inp = out.copy()
            inp[mask] = mask_color
            if _compatible_centers(inp, mask_color) == [(sy, sx)]:
                return {"input": inp, "output": out}
    raise RuntimeError("Could not construct an identifiable pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    mask_color = int(rng.integers(10))
    train_modes = [(0, False), (int(rng.integers(1, 3)), True)]
    train_modes.extend(
        (int(rng.choice(3, p=(0.35, 0.4, 0.25))), bool(rng.random() < 0.45))
        for _ in range(nt - 2)
    )
    rng.shuffle(train_modes)
    test_modes = [
        (int(rng.choice(3, p=(0.2, 0.38, 0.42))), bool(rng.random() < 0.6))
        for _ in range(ns)
    ]
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, modes in (("train", train_modes), ("test", test_modes)):
        for category, hard in modes:
            for _ in range(5):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _pair(pair_rng, mask_color, category, hard)
                inp = pair["input"]
                signature = (inp.shape, inp.tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct pairs")
    return puzzle


def _same_puzzle(a, b):
    return all(
        len(a[section]) == len(b[section])
        and all(
            np.array_equal(x["input"], y["input"])
            and np.array_equal(x["output"], y["output"])
            for x, y in zip(a[section], b[section])
        )
        for section in ("train", "test")
    )


def _component_shapes(mask):
    h, w = mask.shape
    seen = set()
    shapes = []
    for rr, cc in zip(*np.where(mask)):
        start = (int(rr), int(cc))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= y < h and 0 <= x < w and mask[y, x] and (y, x) not in seen:
                    seen.add((y, x))
                    stack.append((y, x))
        box = (
            (max(r for r, c in cells) - min(r for r, c in cells) + 1)
            * (max(c for r, c in cells) - min(c for r, c in cells) + 1)
        )
        shapes.append((len(cells), box))
    return shapes


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    colors = set()
    train_simple = train_indirect = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            h, w = inp.shape
            assert (h - w) % 2 == 0
            key = (inp.shape, inp.tobytes())
            pair_key = (inp.shape, inp.tobytes(), out.tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert hashlib.sha256(inp.astype(np.uint8).tobytes()).hexdigest() not in _SOURCE_INPUT_HASHES
            mask = inp != out
            assert np.any(mask)
            changed = np.unique(inp[mask])
            assert len(changed) == 1
            color = int(changed[0])
            colors.add(color)
            assert not np.any(out == color)
            assert np.array_equal(mask, inp == color)
            assert len(np.unique(out)) >= 4
            centers = _compatible_centers(inp, color)
            assert len(centers) == 1
            sy, sx = centers[0]
            assert _recoverable(mask, sy, sx)
            assert np.array_equal(_restore(inp, color, sy, sx), out)
            for r in range(h):
                for c in range(w):
                    assert all(out[y, x] == out[r, c] for y, x in _orbit(r, c, sy, sx, h, w))
            if section == "train":
                train_simple |= float(mask.mean()) < 0.085 and not _indirect(mask, sy, sx)
                train_indirect |= _indirect(mask, sy, sx)
    assert len(colors) == 1
    assert train_simple and train_indirect


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, shapes, colors, palettes = set(), set(), set(), set()
    centers, areas = set(), set()
    train_hard_positions, test_right_edge = set(), set()
    component_counts, difficulty = set(), set()
    irregular = rectangular = odd = combined = interior_test = 0
    test_indirect = test_easy = 0
    changed_shapes = changed_centers = changed_palettes = 0
    first_outputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_outputs.add(puzzle["train"][0]["output"].tobytes())
        if seed in (0, 7, 21):
            assert _same_puzzle(puzzle, generate_puzzle(seed))
        puzzle_shapes, puzzle_centers, puzzle_palettes = set(), set(), set()
        for section in ("train", "test"):
            for i, pair in enumerate(puzzle[section]):
                inp, out = pair["input"], pair["output"]
                mask = inp != out
                color = int(inp[mask][0])
                sy, sx = _compatible_centers(inp, color)[0]
                sizes.add(inp.shape[0])
                shapes.add(inp.shape)
                colors.add(color)
                palettes.add(tuple(np.unique(out)))
                offset = (sy - inp.shape[0] + 1, sx - inp.shape[1] + 1)
                centers.add(offset)
                areas.add(int(mask.sum()))
                puzzle_shapes.add(inp.shape)
                puzzle_centers.add(offset)
                puzzle_palettes.add(tuple(np.unique(out)))
                parts = _component_shapes(mask)
                component_counts.add(len(parts))
                irregular += any(size != box for size, box in parts)
                rectangular += inp.shape[0] != inp.shape[1]
                odd += inp.shape[0] % 2
                fraction = float(mask.mean())
                level = 0 if fraction < 0.08 else 1 if fraction < 0.15 else 2
                difficulty.add((section, level))
                combined += (
                    inp.shape[0] != inp.shape[1]
                    and any(size != box for size, box in parts)
                    and _indirect(mask, sy, sx)
                )
                assert np.array_equal(_restore(inp, color, sy, sx), out)
                if section == "train" and _indirect(mask, sy, sx):
                    train_hard_positions.add(i)
                if section == "test":
                    test_right_edge.add(bool(mask[:, -1].any()))
                    interior_test += not (
                        mask[0, :].any() or mask[-1, :].any()
                        or mask[:, 0].any() or mask[:, -1].any()
                    )
                    test_indirect += _indirect(mask, sy, sx)
                    test_easy += not _indirect(mask, sy, sx)
        changed_shapes += len(puzzle_shapes) > 1
        changed_centers += len(puzzle_centers) > 1
        changed_palettes += len(puzzle_palettes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 7 and len(shapes) >= 15
    assert len(colors) >= 8 and len(palettes) >= 15
    assert len(centers) >= 9 and len(areas) >= 30
    assert len(train_hard_positions) >= 3 and test_right_edge == {False, True}
    assert len(component_counts) >= 5 and irregular >= 60
    assert rectangular >= 70 and odd >= 50 and combined >= 15
    assert difficulty == {
        ("train", 0), ("train", 1), ("train", 2),
        ("test", 0), ("test", 1), ("test", 2),
    }
    assert interior_test >= 10 and test_indirect >= 10 and test_easy >= 5
    assert min(changed_shapes, changed_centers, changed_palettes) >= 35
    assert len(first_outputs) >= 40
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        a, b = generate_puzzle(9127, nt, ns), generate_puzzle(9127, nt, ns)
        validate_generated_puzzle(a)
        assert _same_puzzle(a, b)
        if nt is not None:
            assert len(a["train"]) == nt
        if ns is not None:
            assert len(a["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(3, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
