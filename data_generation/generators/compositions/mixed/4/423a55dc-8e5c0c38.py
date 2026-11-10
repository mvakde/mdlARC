"""Reflection-supported, globally bottom-anchored left shear.

Only Python's standard library and NumPy are used. Generation constructs clean
cores before adding noise; solve sees only the input. No parent code is imported.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 949199032, 537617516, 136523049, 497879343, 645935044,
              90993165, 32255782, 300311351, 216077352, 63168097, 605576942,
              196377846, 789759610, 769996617, 4284324, 795103069, 284209934,
              28298162, 604883604, 404154003, 498374903, 610272248, 280676613,
              43480178, 245386873, 518879936, 84369477, 404771470, 325233109,
              417047379, 68190870, 247764460, 242760030, 928771411, 14926041,
              817577315, 959870143, 935351659, 311827451, 189490468, 952793807)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= a.size:
        raise ValueError("background must occupy a strict majority")
    return bg


def _scores(mask):
    # Ordered reflected partners: off-axis pairs score two; self-support one.
    scores = np.zeros(2 * mask.shape[1] - 1, dtype=np.int64)
    for row in mask.astype(np.int64):
        scores += np.convolve(row, row)
    return scores


def _filter(a, bg):
    retained = np.full_like(a, bg)
    axes = {}
    cols = np.arange(a.shape[1])
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        mask = a == color
        scores = _scores(mask)
        winners = np.flatnonzero(scores == scores.max())
        if len(winners) != 1:
            raise ValueError("foreground color has tied maximum-support axes")
        axis = int(winners[0])
        reflected = axis - cols
        valid = (0 <= reflected) & (reflected < a.shape[1])
        keep = np.zeros(a.shape, dtype=bool)
        keep[:, valid] = mask[:, valid] & mask[:, reflected[valid]]
        retained[keep] = color
        axes[color] = axis
    return retained, axes


def _shear(a, bg, anchor=None):
    out = np.full_like(a, bg)
    rows, cols = np.nonzero(a != bg)
    if not len(rows):
        return out
    bottom = int(rows.max()) if anchor is None else int(anchor)
    target = cols - (bottom - rows)
    valid = (0 <= target) & (target < a.shape[1])
    out[rows[valid], target[valid]] = a[rows[valid], cols[valid]]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer each color's unique vertical support, then shear all retained rows."""
    a = _grid(input_grid)
    bg = _background(a)
    retained, _ = _filter(a, bg)
    return _shear(retained, bg)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _mask(rng, h, w, axis_tip=False):
    """Symmetric row profiles, holes, contours, disconnected islands and tips."""
    family = int(rng.integers(6))
    mask = np.zeros((h, w), dtype=bool)
    mid = (w - 1) / 2
    left_cols = list(range((w + 1) // 2))
    phase = int(rng.integers(h))
    for r in range(h):
        if family == 0:
            chosen = left_cols
        elif family == 1:
            chosen = left_cols if r in (0, h - 1) else [0]
        elif family in (2, 3):
            t = abs(2 * r - (h - 1)) / max(1, h - 1)
            extent = max(0.5 if w % 2 == 0 else 0,
                         mid * (1 - t if family == 2 else t))
            chosen = [c for c in left_cols if mid - c <= extent + 0.6]
        elif family == 4:
            chosen = [c for c in left_cols if rng.random() < .55]
        else:
            chosen = [(r + phase) % len(left_cols)]
            if rng.random() < .4:
                chosen.append(left_cols[-1])
        if not chosen:
            chosen = [left_cols[-1]]
        for c in chosen:
            mask[r, c] = mask[r, w - 1 - c] = True
    # Attain the full declared width somewhere, not necessarily in the middle.
    row = int(rng.integers(h))
    mask[row, 0] = mask[row, -1] = True
    if w % 2:
        mask[int(rng.integers(h)), w // 2] = True
        if axis_tip:
            tip = 0 if row != 0 else h - 1
            mask[tip] = False
            mask[tip, w // 2] = True
    return mask


def _noise(rng, inp, core, axes, bg, bottom, change_anchor, budget):
    """Add unmatched cells while preserving each intended unique maximum."""
    colors = sorted(axes)

    def add(color, p):
        r, c = p
        axis = axes[color]
        if inp[p] != bg or 2 * c == axis:
            return False
        rc = axis - c
        if 0 <= rc < inp.shape[1] and inp[r, rc] == color:
            return False
        inp[p] = color
        scores = _scores(inp == color)
        if (scores[axis] != np.count_nonzero(core == color)
                or np.count_nonzero(scores == scores[axis]) != 1
                or scores.max() != scores[axis]):
            inp[p] = bg
            return False
        return True

    added = 0
    if change_anchor:
        candidates = [(r, c) for r in range(bottom + 1, inp.shape[0])
                      for c in range(inp.shape[1])]
        for i in rng.permutation(len(candidates)):
            p = candidates[int(i)]
            if add(int(rng.choice(colors)), p):
                added += 1
                break
        if not added:
            return False
    # Prefer additions near a core or in a hole, with some distant specks.
    nearby, distant = [], []
    for r, c in np.argwhere(inp == bg):
        r, c = int(r), int(c)
        if not change_anchor and r > bottom:
            continue
        patch = core[max(0, r-1):r+2, max(0, c-1):c+2]
        (nearby if np.any(patch != bg) else distant).append((r, c))
    rng.shuffle(nearby)
    rng.shuffle(distant)
    candidates = nearby + distant if rng.random() < .8 else distant + nearby
    # Bounded candidate count; no disk journals, retries, or traces are stored.
    for p in candidates[:250]:
        if added >= budget:
            break
        for color in rng.permutation(colors):
            if add(int(color), p):
                added += 1
                break
    return added >= 1


def _make_pair(rng, count, span, gap, clipped, change_anchor, bg,
               informative=False):
    # A gap denotes exactly one interval of absent RETAINED rows.
    active = list(range(span))
    if gap:
        start = int(rng.integers(1, span - gap))
        active = [r for r in active if not start <= r < start + gap]
    width = int(rng.integers(max(14, span + (8 if not clipped else 4)), 31))
    top = int(rng.integers(0, 5))
    bottom = top + span - 1
    margin = int(rng.integers(1 if change_anchor else 0, 7))
    height = min(30, max(bottom + 1 + margin, int(rng.integers(10, 25))))
    if rng.random() < .18:
        height = max(height, width)
    core = np.full((height, width), bg, dtype=np.int8)
    colors = list(map(int, rng.choice([c for c in range(10) if c != bg],
                                     count, replace=False)))
    axes = {}
    first_parity = int(rng.integers(2))
    for index, color in enumerate(colors):
        # The main figure supplies the global endpoints. A shorter second
        # figure ends above the main one, exposing the shared global anchor.
        if index == 0:
            row_indices = active
        elif informative and index == 1:
            stop = int(rng.integers(2, len(active)))
            start = int(rng.integers(0, stop - 1))
            row_indices = active[start:stop]
        else:
            start = int(rng.integers(0, len(active) - 1))
            stop = int(rng.integers(start + 2, len(active) + 1))
            row_indices = active[start:stop]
        parity = (first_parity + index) % 2 if informative else int(rng.integers(2))
        choices = [w for w in range(2, min(10, width) + 1) if (w - 1) % 2 == parity]
        w = int(rng.choice(choices))
        local = _mask(rng, len(row_indices), w, axis_tip=informative and parity == 0)
        rr, cc = np.nonzero(local)
        rows = np.array([top + row_indices[int(r)] for r in rr])
        needed = max(0, int(np.max(bottom - rows - cc)))
        if clipped and index == 0:
            placements = list(range(0, min(needed, width - w + 1)))
        elif not clipped:
            placements = list(range(needed, width - w + 1))
        else:
            placements = list(range(width - w + 1))
        rng.shuffle(placements)
        for left in placements:
            if np.all(core[rows, cc + left] == bg):
                axis = 2 * left + w - 1
                if informative and axis in axes.values():
                    continue
                core[rows, cc + left] = color
                axes[color] = axis
                break
        else:
            return None
    if np.count_nonzero(core != bg) > core.size * .38:
        return None
    inp = core.copy()
    budget = int(rng.integers(1, min(14, 2 + np.count_nonzero(core != bg) // 4)))
    if not _noise(rng, inp, core, axes, bg, bottom, change_anchor, budget):
        return None
    if np.count_nonzero(inp != bg) * 2 >= inp.size:
        return None
    # Ground truth is from the constructed, pre-noise cores, not solve(inp).
    out = np.full_like(core, bg)
    for r, c in np.argwhere(core != bg):
        target = int(c) - (bottom - int(r))
        if target >= 0:
            out[r, target] = core[r, c]
    if clipped != (np.count_nonzero(out != bg) < np.count_nonzero(core != bg)):
        return None
    if np.array_equal(out, core) or np.array_equal(out, _shear(inp, bg)):
        return None
    # Every color remains represented, even if some of its upper rows vanish.
    if set(np.unique(out)) != set(np.unique(inp)):
        return None
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("height", "count", "spacing")))
    fixed_bg = int(rng.integers(10)) if rng.random() < .4 else None
    # Two complementary demonstrations remain present even at count (2,1).
    specs = [(False, True, True), (True, False, False)]
    specs += [(bool(rng.integers(2)), bool(rng.integers(2)), False)
              for _ in range(num_train - 2)]
    rng.shuffle(specs)
    specs += [(bool(rng.integers(2)), bool(rng.integers(2)), False)
              for _ in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for index, (clipped, change_anchor, informative) in enumerate(specs):
        test = index >= num_train
        if shift == "height":
            span = int(rng.integers(8, 12) if test else rng.integers(4, 7))
            gap = int(rng.integers(2))
        elif shift == "spacing":
            gap = int(rng.integers(2, 4)) if test else (0 if informative else 1)
            span = int(rng.integers(4, 7)) + gap
        else:
            span = int(rng.integers(4, 9))
            gap = int(rng.integers(2))
        if shift == "count":
            count = int(rng.integers(3, 5)) if test else (2 if informative else int(rng.integers(1, 3)))
        else:
            count = 2 if informative else int(rng.integers(1, 5))
        # Always teach a physical row gap, for occupied-row-rank shortcuts.
        if not test and clipped:
            gap = max(1, gap)
        bg = fixed_bg if fixed_bg is not None else int(rng.integers(10))
        for _ in range(160):
            pair = _make_pair(rng, count, span, gap, clipped, change_anchor,
                              bg, informative)
            if pair is None or _key(pair["input"]) in seen:
                continue
            # Require concrete observable witnesses in the two anchor examples.
            facts = measure_pair(pair)
            if informative and not all(facts[k] for k in ("per_color_differs", "axis_cells", "different_axes")):
                continue
            if clipped and not test and not facts["rank_differs"]:
                continue
            seen.add(_key(pair["input"]))
            puzzle["test" if test else "train"].append(pair)
            break
        else:
            raise RuntimeError("bounded construction exhausted")
    return puzzle


def _coordinate_filter(a, bg):
    """Validator: explicit coordinate sets, separate from convolution in solve."""
    retained = np.full_like(a, bg)
    axes = {}
    for color in set(map(int, np.unique(a))) - {bg}:
        cells = set(map(tuple, np.argwhere(a == color).tolist()))
        subsets = [{(r, c) for r, c in cells if (r, s - c) in cells}
                   for s in range(2 * a.shape[1] - 1)]
        scores = [len(subset) for subset in subsets]
        winners = [s for s, score in enumerate(scores) if score == max(scores)]
        assert len(winners) == 1, "axis tie"
        axis = winners[0]
        axes[color] = axis
        for p in subsets[axis]:
            retained[p] = color
    return retained, axes


def measure_pair(pair):
    """Bounded, input-derived diagnostics; no latent generation metadata."""
    a, out = _grid(pair["input"]), _grid(pair["output"])
    bg = _background(a)
    core, axes = _filter(a, bg)
    rows, cols = np.nonzero(core != bg)
    bottom, top = int(rows.max()), int(rows.min())
    active = sorted(set(map(int, rows)))
    per_color = np.full_like(a, bg)
    for color in axes:
        color_grid = np.where(core == color, color, bg).astype(np.int8)
        shifted = _shear(color_grid, bg)
        per_color[shifted != bg] = color
    ranked = np.full_like(a, bg)
    for rank, r in enumerate(active):
        distance = len(active) - 1 - rank
        if distance < a.shape[1]:
            ranked[r, :a.shape[1]-distance] = core[r, distance:]
    raw_bottom = int(np.nonzero(a != bg)[0].max())
    wrong_anchor = _shear(core, bg, raw_bottom)
    axis_cells = sum(int(np.count_nonzero(core[:, axis // 2] == color))
                     for color, axis in axes.items() if axis % 2 == 0)
    profiles = []
    for color in sorted(axes):
        mask = core == color
        rr, cc = np.nonzero(mask)
        crop = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]
        profiles.append(hashlib.sha256(bytes(crop.shape) + crop.tobytes()).hexdigest())
    return {"background": bg, "count": len(axes), "span": bottom-top+1,
            "gap": max((b-a-1 for a, b in zip(active, active[1:])), default=0),
            "noise": int(np.count_nonzero(a != core)),
            "clipped": int(np.count_nonzero(core != bg) - np.count_nonzero(out != bg)),
            "anchor_changed": raw_bottom > bottom,
            "wrong_anchor_differs": not np.array_equal(out, wrong_anchor),
            "per_color_differs": not np.array_equal(out, per_color),
            "rank_differs": not np.array_equal(out, ranked),
            "erase_only_differs": not np.array_equal(out, core),
            "shear_all_differs": not np.array_equal(out, _shear(a, bg)),
            "canvas_anchor_differs": not np.array_equal(out, _shear(core, bg, a.shape[0]-1)),
            "different_axes": len(set(axes.values())) > 1,
            "parities": sorted({axis % 2 for axis in axes.values()}),
            "axis_cells": axis_cells, "profiles": profiles}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            bg = _background(a)
            core, axes = _coordinate_filter(a, bg)
            assert 1 <= len(axes) <= 4
            assert set(np.unique(a)) == set(np.unique(out))
            assert np.all((core == a) | (core == bg)), "filter only erases"
            bottom = int(np.nonzero(core != bg)[0].max())
            expected = np.full_like(a, bg)
            survivors = 0
            for r, c in np.argwhere(core != bg):
                target = int(c) - (bottom - int(r))
                if target >= 0:
                    expected[r, target] = core[r, c]
                    survivors += 1
            assert np.array_equal(out, expected), "wrong reflection/shear output"
            assert survivors == np.count_nonzero(out != bg), "row map must be injective"
            assert np.array_equal(out[bottom], core[bottom]), "anchor row must stay fixed"
            assert np.all(out[bottom+1:] == bg)
            assert np.array_equal(solve(a.copy()), out)
            fact = measure_pair(pair)
            assert fact["noise"] > 0
            assert fact["erase_only_differs"] and fact["shear_all_differs"]
            facts[split].append(fact)
    train, test = facts["train"], facts["test"]
    assert any(f["clipped"] > 0 for f in train)
    assert any(f["clipped"] == 0 for f in train)
    for k in ("wrong_anchor_differs", "per_color_differs", "rank_differs",
              "canvas_anchor_differs", "different_axes", "axis_cells"):
        assert any(f[k] for f in train), f"missing training witness: {k}"
    taught = set(p for f in train for p in f["parities"])
    assert set(p for f in test for p in f["parities"]) <= taught
    # Observable shift, no hidden split label is required by solve.
    assert any(min(f[k] for f in test) > max(f[k] for f in train)
               for k in ("span", "count", "gap")), "missing structural train/test shift"


def self_test():
    """42 host seeds, exact semantic fixtures, mutations, counts, no disk writes."""
    shapes, profiles, backgrounds, counts, parities = set(), set(), set(), set(), set()
    fingerprints, shift_kinds = set(), set()
    clipped_tests, anchor_tests = set(), set()
    total_pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        assert all(_key(p[k]) == _key(q[k])
                   for s in puzzle for p, q in zip(puzzle[s], again[s])
                   for k in ("input", "output"))
        fingerprints.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        split_facts = {s: [measure_pair(p) for p in puzzle[s]] for s in puzzle}
        for k in ("span", "count", "gap"):
            if min(f[k] for f in split_facts["test"]) > max(f[k] for f in split_facts["train"]):
                shift_kinds.add(k)
        for s in puzzle:
            for pair, fact in zip(puzzle[s], split_facts[s]):
                total_pairs += 1
                shapes.add(pair["input"].shape)
                profiles.update(fact["profiles"])
                backgrounds.add(fact["background"])
                counts.add(fact["count"])
                parities.update(fact["parities"])
                if s == "test":
                    clipped_tests.add(fact["clipped"] > 0)
                    anchor_tests.add(fact["anchor_changed"])
    assert len(fingerprints) == 42
    assert len(shapes) >= 50 and len(profiles) >= 100
    assert backgrounds == set(range(10)) and counts == {1, 2, 3, 4}
    assert parities == {0, 1} and shift_kinds == {"span", "count", "gap"}
    assert clipped_tests == anchor_tests == {False, True}
    # Explicit fixture: unequal per-color bottoms, a row gap, axis self-support,
    # unsupported bottom noise, clipping, and a surviving isolated paired island.
    a = np.zeros((8, 13), dtype=np.int8)
    core = a.copy()
    for r, columns in ((0, (1, 5)), (2, (1, 2, 3, 4, 5)), (4, (3,))):
        for c in columns:
            a[r, c] = core[r, c] = 2
    for r in (0, 2):
        a[r, 9:11] = core[r, 9:11] = 6
    a[7, 0] = 2
    expected = np.zeros_like(a)
    expected[0, 1] = expected[4, 3] = 2
    expected[2, :4] = 2
    expected[0, 5:7] = expected[2, 7:9] = 6
    before = a.copy()
    assert np.array_equal(solve(a), expected)
    assert np.array_equal(a, before), "solve must not mutate its input"
    assert expected[0, 1] == 2 and expected[4, 3] == 2
    assert np.array_equal(expected[0, 5:7], [6, 6])
    # Ties are outside the domain; do not silently impose a leftmost preference.
    tied = np.zeros((3, 7), dtype=np.int8)
    tied[1, 1] = tied[2, 5] = 2
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous axes accepted")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            validate_generated_puzzle(generate_puzzle(173 + tr * 7 + te, tr, te))
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(31, 2, 1)
    out = bad["test"][0]["output"]
    bg = _background(bad["test"][0]["input"])
    r, c = np.argwhere(out != bg)[0]
    out[r, c] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a deleted survivor")
    return {"seeds": 42, "pairs": total_pairs, "shapes": len(shapes),
            "profiles": len(profiles), "backgrounds": len(backgrounds),
            "counts": sorted(counts), "shift_kinds": sorted(shift_kinds)}
