"""Reflection-supported foreground, followed by the fixed bottom-anchored shear.

No parent module, file, seed, or example is consulted by solve(). All checks and
experiments stream small puzzles in memory; this module writes no artifacts.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid of colors 0..9")
    return a.astype(np.int8, copy=False)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if 2 * counts[bg] <= a.size:
        raise ValueError("input background must occupy a strict majority")
    return bg


def _clean(a):
    """Convolution counts ordered same-row pairs, including self-pairs."""
    bg = _background(a)
    cleaned = np.full_like(a, bg)
    axes = {}
    columns = np.arange(a.shape[1])
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        mask = (a == color).astype(np.int16)
        scores = np.zeros(2 * a.shape[1] - 1, dtype=np.int32)
        for row in mask:
            scores += np.convolve(row, row)
        winners = np.flatnonzero(scores == scores.max())
        if len(winners) != 1:
            raise ValueError("foreground color has tied maximum-support axes")
        axis = int(winners[0])
        reflected = axis - columns
        valid = (reflected >= 0) & (reflected < a.shape[1])
        keep = np.zeros(a.shape, dtype=bool)
        keep[:, valid] = (mask[:, valid] != 0) & (mask[:, reflected[valid]] != 0)
        cleaned[keep] = color
        axes[color] = axis
    return cleaned, bg, axes


def _shear(cleaned, bg):
    h, w = cleaned.shape
    out = np.full((h, h + w - 1), bg, dtype=np.int8)
    for r in range(h):
        start = h - 1 - r
        out[r, start:start + w] = cleaned[r]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Delete unsupported cells in the original frame; then shear whole rows."""
    a = _grid(input_grid)
    if sum(a.shape) - 1 > 30:
        raise ValueError("sheared output would exceed 30 columns")
    cleaned, bg, _ = _clean(a)
    return _shear(cleaned, bg)


def _cells(a, color):
    return set(map(tuple, np.argwhere(a == color).tolist()))


def _supported(cells, width):
    """Coordinate-set construction/validation, separate from solve's convolution."""
    scores = [(axis, {p for p in cells if (p[0], axis - p[1]) in cells})
              for axis in range(2 * width - 1)]
    maximum = max(len(s) for _, s in scores)
    winners = [(axis, s) for axis, s in scores if len(s) == maximum]
    if len(winners) != 1:
        raise ValueError("ambiguous axis")
    return winners[0]


def _neighbors(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = remaining.pop()
        component, frontier = {start}, {start}
        while frontier:
            frontier = _neighbors(frontier) & remaining
            remaining -= frontier
            component |= frontier
        result.append(component)
    return result


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _mask(rng, height, width, islands=False):
    """Mirrored row profiles, contours, bars, holes, stems and supported islands."""
    style = int(rng.integers(6))
    mask = np.zeros((height, width), dtype=bool)
    center = (width - 1) / 2
    if style == 0:  # hollow rectangle, occasionally thick caps
        mask[:, 0] = mask[:, -1] = True
        mask[0] = mask[-1] = True
    else:
        levels = rng.integers(1, (width + 1) // 2 + 1, size=height)
        for r in range(height):
            if style == 1:  # pointed profile
                fraction = 1 - abs(2 * r - height + 1) / max(1, height)
                level = max(1, int(round(fraction * ((width + 1) // 2))))
            elif style == 2:  # hourglass
                fraction = abs(2 * r - height + 1) / max(1, height)
                level = max(1, int(round(fraction * ((width + 1) // 2))))
            else:
                level = int(levels[r])
            for c in range((width + 1) // 2):
                inside = abs(c - center) < level
                if inside and (style != 4 or rng.random() < .55):
                    mask[r, c] = mask[r, width - 1 - c] = True
            if style == 5 and width % 2 and rng.random() < .45:
                mask[r] = False
                mask[r, width // 2] = True
    # Guarantee real width; height may include entirely empty rows.
    mask[int(rng.integers(height)), 0] = True
    mask |= mask[:, ::-1]
    if islands and height >= 4:
        gap = int(rng.integers(1, height - 1))
        mask[gap] = False
        mask[0, 0] = mask[0, -1] = True
        mask[-1, 0] = mask[-1, -1] = True
    return mask


def _shape(rng, role, shift):
    if role == "test" and shift == "height":
        h = int(rng.integers(16, 25))
        return h, int(rng.integers(5, min(12, 31 - h) + 1))
    if role == "test" and shift == "width":
        h = int(rng.integers(4, 7))
        return h, int(rng.integers(18, 32 - h))
    if role == "test":
        h = int(rng.integers(11, 17))
        return h, int(rng.integers(11, min(18, 31 - h) + 1))
    h = int(rng.integers(8, 15))
    return h, int(rng.integers(8, min(17, 31 - h) + 1))


def _place(rng, shape, count, anchor):
    h, w = shape
    occupied, figures = set(), []
    for i in range(count):
        for attempt in range(70):
            parity = anchor if i == 0 and anchor is not None else int(rng.integers(2))
            max_w = min(w - 1, 9 if count <= 2 else 6)
            widths = [v for v in range(2, max_w + 1) if (v - 1) % 2 == parity]
            ww = int(rng.choice(widths))
            hh = int(rng.integers(4 if i == 0 and anchor is not None else 1,
                                   min(h - 1, 9 if count <= 2 else 5) + 1))
            mask = _mask(rng, hh, ww, islands=(i == 0 and anchor == 1))
            local = set(map(tuple, np.argwhere(mask).tolist()))
            if len(local) < 4 or len(occupied) + len(local) > .36 * h * w:
                continue
            top = int(rng.integers(h - hh + 1))
            left = int(rng.integers(w - ww + 1))
            core = {(r + top, c + left) for r, c in local}
            if core & occupied:
                continue
            # Usually leave space, but overlapping boxes and touching colors are legal.
            if attempt < 35 and core & _neighbors(occupied):
                continue
            occupied |= core
            figures.append((core, 2 * left + ww - 1))
            break
        else:
            return None
    return figures


def _noise(rng, core, axis, occupied, shape, amount, protrusion=False):
    h, w = shape
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    candidates = {(r, c) for r in range(h) for c in range(w)} - occupied
    candidates = {p for p in candidates if 2 * p[1] != axis
                  and (p[0], axis - p[1]) not in core}
    added = set()
    for step in range(amount):
        near = _neighbors(core | added)
        if step == 0 and protrusion:
            pool = {p for p in candidates & near if p[1] < c0 or p[1] > c1}
        else:
            pool = candidates & near if rng.random() < .6 else candidates
        choices = sorted(pool or (set() if step == 0 and protrusion else candidates))
        accepted = False
        for j in rng.permutation(len(choices))[:40]:
            p = choices[int(j)]
            candidates.discard(p)
            try:
                winner, support = _supported(core | added | {p}, w)
            except ValueError:
                continue
            if winner != axis or support != core:
                continue
            added.add(p)
            candidates.discard((p[0], axis - p[1]))
            accepted = True
            break
        if not accepted:
            break
    return added


def _witnesses(inp, cleaned, bg, axes):
    bbox_wrong = attached = islands = moving = False
    demonstrated = set()
    for color, axis in axes.items():
        original, core = _cells(inp, color), _cells(cleaned, color)
        noise = original - core
        if noise:
            demonstrated.add(axis % 2)
        bbox_axis = min(c for _, c in original) + max(c for _, c in original)
        at_bbox = {p for p in original if (p[0], bbox_axis - p[1]) in original}
        bbox_wrong |= at_bbox != core
        attached |= bool(noise & _neighbors(core))
        # At least two original input components contain cells that must survive.
        # This specifically defeats keeping only the largest connected component.
        islands |= sum(bool(component & core) for component in _components(original)) > 1
        moving |= len({r for r, _ in core}) > 1
    return {"bbox_wrong": bbox_wrong, "attached": attached, "islands": islands,
            "moving": moving, "demonstrated": demonstrated}


def _random_pair(rng, role, shift, count, anchor, bg):
    for _ in range(60):
        shape = _shape(rng, role, shift)
        figures = _place(rng, shape, count, anchor)
        if figures is None:
            continue
        colors = list(map(int, rng.choice([c for c in range(10) if c != bg], count, replace=False)))
        inp = np.full(shape, bg, dtype=np.int8)
        cleaned = inp.copy()
        occupied = set().union(*(core for core, _ in figures))
        for color, (core, _) in zip(colors, figures):
            for p in core:
                inp[p] = cleaned[p] = color
        unchanged = anchor is None and role == "train" and rng.random() < .12
        for i, (color, (core, axis)) in enumerate(zip(colors, figures)):
            clean = unchanged or (i > 0 or anchor is None) and rng.random() < .25
            amount = 0 if clean else min(12, max(1, int(round(len(core) * rng.uniform(.10, .45)))))
            amount = min(amount, max(0, int(inp.size * .47) - len(occupied)))
            added = _noise(rng, core, axis, occupied, shape, amount,
                           protrusion=(anchor == 0 and i == 0))
            occupied |= added
            for p in added:
                inp[p] = color
        axes = dict(zip(colors, (a for _, a in figures)))
        witness = _witnesses(inp, cleaned, bg, axes)
        if anchor is not None:
            first_noise = _cells(inp, colors[0]) - figures[0][0]
            if not first_noise or not witness["moving"]:
                continue
            if anchor == 0 and not (witness["bbox_wrong"] and witness["attached"]):
                continue
            if anchor == 0:
                try:
                    horizontal, _, _ = _clean(inp.T)
                    if np.array_equal(horizontal.T, cleaned):
                        continue
                except ValueError:
                    pass
            if anchor == 1 and (not witness["islands"] or len(set(axes.values())) < 2):
                continue
        if role == "test" and np.array_equal(inp, cleaned):
            continue
        return {"input": inp, "output": _shear(cleaned, bg)}
    return _fallback_pair(rng, role, shift, count, anchor, bg)


def _fallback_pair(rng, role, shift, count, anchor, bg):
    """Bounded, explicit small geometry; never chains a parent generator."""
    if role == "test":
        shape = {"height": (20, 9), "width": (5, 25), "count": (14, 16)}[shift]
    else:
        shape = (10, 15) if anchor != 1 else (11, 14)
    inp = np.full(shape, bg, dtype=np.int8)
    cleaned = inp.copy()
    colors = rng.choice([c for c in range(10) if c != bg], count, replace=False)
    # Slots are only the guaranteed fallback, not the normal placement policy.
    slots = [(r, c) for r in range(0, shape[0] - 3, 4)
             for c in range(0, shape[1] - 4, 5)]
    assert len(slots) >= count
    for i, (color, (r, c)) in enumerate(zip(colors, slots)):
        parity = anchor if i == 0 and anchor is not None else int(rng.integers(2))
        ww = 3 if parity == 0 else 4
        core = {(rr, cc) for rr in (r, r + 1, r + 2) for cc in range(c, c + ww)}
        if anchor == 0 and i == 0:
            core.add((r + 3, c + 1))
        if anchor == 1 and i == 0:
            core = {(rr, cc) for rr in (r, r + 2, r + 3) for cc in range(c, c + ww)}
        for p in core:
            inp[p] = cleaned[p] = color
        inp[r, c + ww] = color
        axis, supported = _supported(_cells(inp, int(color)), shape[1])
        assert axis == 2 * c + ww - 1 and supported == core
    return {"input": inp, "output": _shear(cleaned, bg)}


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(("height", "width", "count")))
    puzzle, seen = {"train": [], "test": []}, set()
    first_bg = None
    for role, length in (("train", num_train), ("test", num_test)):
        for index in range(length):
            anchor = index if role == "train" and index < 2 else None
            count = (int(rng.integers(1, 3)) if anchor == 0 else
                     int(rng.integers(2, 4)) if anchor == 1 else
                     int(rng.integers(4, 6)) if role == "test" and shift == "count" else
                     int(rng.integers(1, 4)))
            pool = [c for c in range(10) if not (anchor == 1 and c == first_bg)]
            bg = int(rng.choice(pool))
            for attempt in range(20):
                pair = _random_pair(rng, role, shift, count, anchor, bg)
                a = pair["input"]
                if _key(a) in seen:
                    continue
                if anchor == 1 and a.shape[0] == puzzle["train"][0]["input"].shape[0]:
                    continue
                break
            else:
                raise RuntimeError("bounded unique-example sampling exhausted")
            seen.add(_key(a))
            puzzle[role].append(pair)
            if anchor == 0:
                first_bg = bg
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, taught, records = set(), set(), {"train": [], "test": []}
    train_witnesses = []
    horizontal_rejected = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            h, w = a.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            assert out.shape == (h, h + w - 1)
            bg = _background(a)
            cleaned = np.empty_like(a)
            for r in range(h):
                start = h - 1 - r
                cleaned[r] = out[r, start:start + w]
                assert np.all(out[r, :start] == bg) and np.all(out[r, start + w:] == bg)
            assert np.all((cleaned == a) | (cleaned == bg)), "cleaning may only delete"
            colors = set(map(int, np.unique(a))) - {bg}
            assert 1 <= len(colors) <= 5
            axes = {}
            for color in colors:
                axis, support = _supported(_cells(a, color), w)
                assert support == _cells(cleaned, color), "wrong reflected subset"
                assert any(2 * c != axis for _, c in support), "generated cores need off-axis evidence"
                axes[color] = axis
            assert np.array_equal(solve(a.copy()), out)
            witness = _witnesses(a, cleaned, bg, axes)
            if split == "train":
                taught |= witness["demonstrated"]
                train_witnesses.append(witness)
                try:
                    horizontal, _, _ = _clean(a.T)
                    horizontal_rejected |= not np.array_equal(horizontal.T, cleaned)
                except ValueError:
                    horizontal_rejected = True
                if len(train_witnesses) == 2:
                    assert len(set(axes.values())) > 1, "demonstrate independent color axes"
            else:
                assert not np.array_equal(a, cleaned), "query must exercise cleaning"
                assert {axis % 2 for axis in axes.values()} <= taught
            # Explicit omission tests: raw shear carries outliers; omitted shear has wrong dimensions.
            if witness["demonstrated"]:
                assert not np.array_equal(_shear(a, bg), out)
            assert h > 1 and out.shape != cleaned.shape
            records[split].append((h, w, len(colors), bg))
    assert taught == {0, 1}
    assert horizontal_rejected, "demonstrate vertical rather than horizontal reflection"
    assert train_witnesses[0]["bbox_wrong"] and train_witnesses[0]["attached"]
    assert train_witnesses[1]["islands"]
    assert all(v["moving"] for v in train_witnesses[:2])
    assert records["train"][0][0] != records["train"][1][0]
    assert records["train"][0][3] != records["train"][1][3]
    # The split shift is visible, and concerns one of the actual mechanisms.
    for test in records["test"]:
        assert any(test[d] > max(t[d] for t in records["train"]) for d in range(3))


def self_test():
    """42 streaming seeds; coverage, coordinate semantics and adversarial mutations."""
    shapes, palettes, backgrounds, counts, parity_mixes, signatures = set(), set(), set(), set(), set(), set()
    metrics = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                signature.update(bytes(a.shape) + a.tobytes() + out.tobytes())
                shapes.add(a.shape)
                bg = _background(a)
                backgrounds.add(bg)
                palettes.add(tuple(map(int, np.unique(a))))
                cleaned, _, axes = _clean(a)
                counts.add(len(axes))
                parity_mixes.add(tuple(sorted({v % 2 for v in axes.values()})))
                metrics["boundary"] += out.shape[1] == 30
                metrics["clean_pairs"] += np.array_equal(a, cleaned)
                for color, axis in axes.items():
                    core, original = _cells(cleaned, color), _cells(a, color)
                    assert _supported(original, a.shape[1]) == (axis, core)
                    metrics["clean_figures"] += core == original
                    metrics["islands"] += len(_components(core)) > 1
                    metrics["edge"] += any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1) for r, c in core)
                    metrics["axis_only"] += any(sum(rr == r for rr, _ in core) == 1 for r, _ in core)
                    metrics["attached"] += bool((original - core) & _neighbors(core))
                    metrics["detached"] += bool((original - core) - _neighbors(core))
                    # Every surviving source coordinate has exactly its row's prescribed destination.
                    assert all(out[r, c + a.shape[0] - 1 - r] == color for r, c in core)
        signatures.add(signature.digest())
        if seed < 2:
            other = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], other[s]) for k in p)
    assert len(signatures) == 42 and len(shapes) >= 45 and len(palettes) >= 65
    assert backgrounds == set(range(10)) and counts == set(range(1, 6))
    assert parity_mixes == {(0,), (1,), (0, 1)}
    assert all(metrics[name] > 0 for name in ("boundary", "clean_pairs", "clean_figures", "islands",
                                             "edge", "axis_only", "attached", "detached"))
    # Exercise guaranteed fallbacks explicitly, including their demonstration contracts.
    for anchor, n in ((0, 1), (0, 2), (1, 2), (1, 3)):
        pair = _fallback_pair(np.random.default_rng(40 + n), "train", "height", n, anchor, 0)
        a = pair["input"]
        cleaned, bg, axes = _clean(a)
        assert np.array_equal(solve(a), pair["output"])
        witness = _witnesses(a, cleaned, bg, axes)
        assert witness["moving"]
        if anchor == 0:
            assert witness["bbox_wrong"] and witness["attached"]
            try:
                horizontal, _, _ = _clean(a.T)
                assert not np.array_equal(horizontal.T, cleaned)
            except ValueError:
                pass
        else:
            assert witness["islands"]
    for shift, n in (("height", 3), ("width", 3), ("count", 5)):
        pair = _fallback_pair(np.random.default_rng(50), "test", shift, n, None, 7)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    # A tiny hand-computed example: original axis 2; a right-hand addition is erased.
    example = np.array([[0, 0, 0, 0, 0], [0, 2, 2, 2, 2],
                        [0, 2, 0, 2, 0], [0, 2, 2, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int8)
    expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 2, 0, 0],
                         [0, 0, 0, 2, 0, 2, 0, 0, 0], [0, 0, 2, 2, 2, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert np.array_equal(solve(example), expected)
    one_row = np.array([[0, 1, 1, 0, 0]], dtype=np.int8)
    assert np.array_equal(solve(one_row), one_row)
    one_column = np.zeros((30, 1), dtype=np.int8)
    one_column[[0, 17], 0] = 3
    column_expected = np.zeros((30, 30), dtype=np.int8)
    column_expected[0, 29] = column_expected[17, 12] = 3
    assert np.array_equal(solve(one_column), column_expected)
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(1000 + 10 * n + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    # Reject both lost support and an outlier incorrectly transported into the output.
    for mutation in ("lost_support", "retained_noise", "wrong_padding"):
        p = generate_puzzle(903, 2, 1)
        a, out = p["train"][0]["input"], p["train"][0]["output"]
        cleaned, bg, _ = _clean(a)
        if mutation == "wrong_padding":
            out[0, 0] = (bg + 1) % 10
        else:
            eligible = cleaned != bg if mutation == "lost_support" else a != cleaned
            r, c = np.argwhere(eligible)[0]
            out[r, c + a.shape[0] - 1 - r] = bg if mutation == "lost_support" else a[r, c]
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted corrupted output")
    # Unsupported ties have no implicit leftmost or integral-axis preference.
    try:
        solve(np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int8))
    except ValueError:
        raise AssertionError("a supported two-cell pair has a unique axis")
    try:
        solve(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous axis accepted")
    return {"seeds_checked": 42, "pairs_checked": 210, "input_shapes": len(shapes),
            "palettes": len(palettes), "backgrounds": sorted(backgrounds),
            "figure_counts": sorted(counts), "features": dict(metrics)}
