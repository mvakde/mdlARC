"""Triangular self-substitution ("count-up Kronecker") ARC family.

Rule (input -> output):
  * The input is an h x w two-colour motif.  Its strictly most frequent colour is
    the background b; the other colour is the active colour c.
  * X is the background mask of the motif (B's role inversion: X marks the cells
    that become "ink" in the stamped tiles).
  * The output is T(h) x T(w), T(n) = n(n+1)/2.  Its rows are cut into adjacent
    bands of heights 1, 2, ..., h and its columns into bands of widths 1, ..., w
    (A's count-up partition with triangular offsets, origin at the top-left).
  * Block (i, j) (row band i, column band j, size (i+1) x (j+1)) is filled iff
    X[i, j] is set; it then shows the top-left (i+1) x (j+1) window of X drawn in
    colour c on b.  All other cells are b.
  Equivalently cell (band i, offset a; band j, offset b) is active iff
  X[i, j] and X[a, b] -- B's self-Kronecker formula restricted to a <= i, b <= j.
"""
from __future__ import annotations

import numpy as np

MIN_DIM, MAX_DIM = 2, 7  # T(7) = 28 <= 30


def tri(n: int) -> int:
    return n * (n + 1) // 2


# ---------------------------------------------------------------------------
# Solver (input only)
# ---------------------------------------------------------------------------

def _roles(grid: np.ndarray):
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    bg = int(values[order[0]])
    active = int(values[order[1]]) if len(values) > 1 else bg
    return bg, active


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    h, w = grid.shape
    bg, active = _roles(grid)
    mask = grid == bg
    out = np.full((tri(h), tri(w)), bg, dtype=int)
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                window = mask[: i + 1, : j + 1]
                out[tri(i): tri(i) + i + 1, tri(j): tri(j) + j + 1] = np.where(window, active, bg)
    return out


# ---------------------------------------------------------------------------
# Alternative (shortcut) renderings used to make demonstrations discriminating
# ---------------------------------------------------------------------------

def _render(mask, bg, active, *, place="bg", content="prefix", all_blocks=False):
    h, w = mask.shape
    out = np.full((tri(h), tri(w)), bg, dtype=int)
    for i in range(h):
        for j in range(w):
            present = mask[i, j] if place == "bg" else not mask[i, j]
            if not (present or all_blocks):
                continue
            if content == "solid":
                win = np.ones((i + 1, j + 1), dtype=bool)
            elif content == "prefix":
                win = mask[: i + 1, : j + 1]
            elif content == "rowsuffix":
                win = mask[h - 1 - i:, : j + 1]
            elif content == "colsuffix":
                win = mask[: i + 1, w - 1 - j:]
            elif content == "suffix":
                win = mask[h - 1 - i:, w - 1 - j:]
            elif content == "uninverted":
                win = ~mask[: i + 1, : j + 1]
            else:  # pragma: no cover
                raise ValueError(content)
            out[tri(i): tri(i) + i + 1, tri(j): tri(j) + j + 1] = np.where(win, active, bg)
    return out


ALTERNATIVES = {
    "solid_blocks": dict(content="solid"),
    "row_suffix_window": dict(content="rowsuffix"),
    "col_suffix_window": dict(content="colsuffix"),
    "far_corner_window": dict(content="suffix"),
    "uninverted_window": dict(content="uninverted"),
    "every_block": dict(all_blocks=True),
    "active_placement": dict(place="active"),
}


def _alt_outputs(grid):
    grid = np.asarray(grid, dtype=int)
    bg, active = _roles(grid)
    mask = grid == bg
    return {name: _render(mask, bg, active, **kw) for name, kw in ALTERNATIVES.items()}


# ---------------------------------------------------------------------------
# Motif sampling
# ---------------------------------------------------------------------------

def _density_range(h, w):
    n = h * w
    lo = n // 2 + 1
    hi = max(lo, int(round(0.68 * n)))
    return lo, min(hi, n - 1)


def _pair_ok(mask: np.ndarray) -> bool:
    h, w = mask.shape
    lo, hi = _density_range(h, w)
    k = int(mask.sum())
    if not lo <= k <= hi:
        return False
    if not mask[h - 1].any() or not mask[:, w - 1].any():
        return False
    # Some present block must show a background hole (content matters) ...
    holes = False
    # ... and some absent block's window must contain ink (placement matters).
    absent_ink = False
    for i in range(h):
        for j in range(w):
            win = mask[: i + 1, : j + 1]
            if mask[i, j] and not win.all():
                holes = True
            if not mask[i, j] and win.any():
                absent_ink = True
    return holes and absent_ink


def _canon(mask: np.ndarray):
    forms = []
    m = mask.astype(np.uint8)
    for k in range(4):
        r = np.rot90(m, k)
        for f in (r, r.T):
            forms.append((f.shape, f.tobytes()))
    return min(forms)


def _sample_mask(rng, h, w, kind):
    lo, hi = _density_range(h, w)
    for _ in range(400):
        k = int(rng.integers(lo, hi + 1))
        if kind == "mirror" and w >= 2:
            half = (w + 1) // 2
            left = rng.random((h, half)) < k / (h * w)
            mask = np.concatenate([left, left[:, : w // 2][:, ::-1]], axis=1)
        elif kind == "line":
            mask = np.zeros(h * w, dtype=bool)
            mask[rng.choice(h * w, size=k, replace=False)] = True
            mask = mask.reshape(h, w)
            if rng.random() < 0.5:
                mask[int(rng.integers(0, h - 1)), :] = False  # full active row
            else:
                mask[:, int(rng.integers(0, w - 1))] = False  # full active column
        elif kind == "diagonal":
            mask = np.zeros(h * w, dtype=bool)
            mask[rng.choice(h * w, size=k, replace=False)] = True
            mask = mask.reshape(h, w)
            d = min(h, w)
            if rng.random() < 0.5:
                for t in range(d):
                    mask[t, t] = True
            else:
                for t in range(d):
                    mask[t, d - 1 - t] = True
        else:  # random
            mask = np.zeros(h * w, dtype=bool)
            mask[rng.choice(h * w, size=k, replace=False)] = True
            mask = mask.reshape(h, w)
        if _pair_ok(mask):
            return mask
    # fallback: plain random
    for _ in range(2000):
        k = int(rng.integers(lo, hi + 1))
        mask = np.zeros(h * w, dtype=bool)
        mask[rng.choice(h * w, size=k, replace=False)] = True
        mask = mask.reshape(h, w)
        if _pair_ok(mask):
            return mask
    raise RuntimeError(f"cannot sample mask {h}x{w}")


KINDS = ("random", "random", "mirror", "line", "diagonal")


# ---------------------------------------------------------------------------
# Puzzle generation
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError(f"illegal counts num_train={num_train} num_test={num_test}")


def _shape_pool(max_dim, square_only, min_dim=MIN_DIM):
    pool = []
    for h in range(min_dim, max_dim + 1):
        for w in range(min_dim, max_dim + 1):
            if square_only and h != w:
                continue
            pool.append((h, w))
    return pool


def _plan(rng, num_train, num_test):
    regime = "extrapolate" if rng.random() < 0.5 else "interpolate"
    square_only = rng.random() < 0.3
    if regime == "extrapolate":
        dt = int(rng.choice([3, 4, 5], p=[0.35, 0.45, 0.2]))
    else:
        dt = int(rng.choice([3, 4, 5, 6], p=[0.25, 0.4, 0.25, 0.1]))
    pool = _shape_pool(dt, square_only)
    # training sizes: must include max dim dt, a pair with both dims >= 3,
    # and at least two distinct shapes
    for _ in range(200):
        shapes = []
        weights = np.array([1.0 + (h + w) / 4.0 for h, w in pool])
        weights /= weights.sum()
        for _ in range(num_train):
            shapes.append(pool[int(rng.choice(len(pool), p=weights))])
        if 2 <= len(pool):
            if len(set(shapes)) < 2:
                continue
        if max(max(s) for s in shapes) != dt:
            continue
        if not any(min(s) >= 3 for s in shapes):
            continue
        # avoid too many 2x2 (only one dihedral class exists)
        if sum(1 for s in shapes if s == (2, 2)) > 1:
            continue
        break
    else:
        raise RuntimeError("cannot plan training sizes")
    tests = []
    for t in range(num_test):
        if regime == "extrapolate" and (t == 0 or rng.random() < 0.5):
            big = int(rng.integers(dt + 1, min(MAX_DIM, dt + 2) + 1))
            if square_only or rng.random() < 0.4:
                tests.append((big, big))
            else:
                other = int(rng.integers(3, MAX_DIM + 1))
                tests.append((big, other) if rng.random() < 0.5 else (other, big))
        else:
            cand = [s for s in _shape_pool(dt, square_only, 3)]
            tests.append(cand[int(rng.integers(len(cand)))])
    return regime, square_only, dt, shapes, tests


def _colors(rng, n_pairs):
    bg_mode = "shared" if rng.random() < 0.8 else "varied"
    if bg_mode == "shared":
        bg = int(rng.integers(0, 10))
        others = [c for c in range(10) if c != bg]
        palette = list(rng.choice(others, size=int(rng.integers(1, 5)), replace=False))
        out = []
        for _ in range(n_pairs):
            if rng.random() < 0.25:
                act = int(rng.choice(others))
            else:
                act = int(rng.choice(palette))
            out.append((bg, act))
        return bg_mode, out
    out = []
    for _ in range(n_pairs):
        bg = int(rng.integers(0, 10))
        act = int(rng.choice([c for c in range(10) if c != bg]))
        out.append((bg, act))
    return bg_mode, out


def _make_pair(mask, bg, act):
    grid = np.where(mask, bg, act).astype(int)
    return {"input": grid, "output": solve(grid)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _attempt in range(100):
        regime, square_only, dt, train_shapes, test_shapes = _plan(rng, num_train, num_test)
        bg_mode, colors = _colors(rng, num_train + num_test)
        seen = set()
        masks = []
        ok = True
        for shape in train_shapes + test_shapes:
            for _ in range(60):
                kind = KINDS[int(rng.integers(len(KINDS)))]
                m = _sample_mask(rng, shape[0], shape[1], kind)
                c = _canon(m)
                if c not in seen:
                    seen.add(c)
                    masks.append(m)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        pairs = [_make_pair(m, *col) for m, col in zip(masks, colors)]
        train, test = pairs[:num_train], pairs[num_train:]
        puzzle = {"train": train, "test": test}
        try:
            _check_discrimination(puzzle)
        except AssertionError:
            continue
        for p in train + test:
            p["input"] = p["input"].tolist()
            p["output"] = p["output"].tolist()
        return puzzle
    raise RuntimeError(f"generation failed for seed {seed}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _expected_cellwise(grid):
    """Independent cell-formula oracle (not block slicing)."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    vals, cnts = np.unique(grid, return_counts=True)
    assert len(vals) == 2, "input must have exactly two colours"
    assert cnts[0] != cnts[1], "background must be a strict majority"
    bg = int(vals[np.argmax(cnts)])
    act = int(vals[np.argmin(cnts)])
    band_r = [(i, a) for i in range(h) for a in range(i + 1)]
    band_c = [(j, b) for j in range(w) for b in range(j + 1)]
    out = np.full((len(band_r), len(band_c)), bg, dtype=int)
    for r, (i, a) in enumerate(band_r):
        for c, (j, b) in enumerate(band_c):
            if grid[i, j] == bg and grid[a, b] == bg:
                out[r, c] = act
    return out


def _check_discrimination(puzzle):
    train = puzzle["train"]
    tests = puzzle["test"]
    for name in ALTERNATIVES:
        assert any(not np.array_equal(_alt_outputs(p["input"])[name], np.asarray(p["output"]))
                   for p in train), f"training does not refute shortcut {name}"
    for p in tests:
        alts = _alt_outputs(p["input"])
        for name in ("solid_blocks", "row_suffix_window", "col_suffix_window",
                     "far_corner_window", "every_block"):
            assert not np.array_equal(alts[name], np.asarray(p["output"])), \
                f"test answer coincides with shortcut {name}"
    shapes = {np.asarray(p["input"]).shape for p in train}
    assert len(shapes) >= 2, "training must show at least two motif sizes"
    assert any(min(s) >= 3 for s in shapes), "training needs a motif with both sides >= 3"


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_classes = set(), set()
    for p in train + test:
        assert set(p) == {"input", "output"}
        gi = np.asarray(p["input"], dtype=int)
        go = np.asarray(p["output"], dtype=int)
        assert gi.ndim == 2 and go.ndim == 2
        h, w = gi.shape
        assert MIN_DIM <= h <= MAX_DIM and MIN_DIM <= w <= MAX_DIM
        assert go.shape == (tri(h), tri(w)) and max(go.shape) <= 30
        assert gi.min() >= 0 and gi.max() <= 9
        assert np.array_equal(go, _expected_cellwise(gi)), "output violates cell formula"
        assert np.array_equal(go, solve(gi)), "solve disagrees with cell formula"
        assert set(np.unique(go)) <= set(np.unique(gi))
        bg, _ = _roles(gi)
        mask = gi == bg
        assert _pair_ok(mask), "pair constraints violated"
        key = (gi.shape, gi.tobytes())
        assert key not in seen_inputs, "duplicate input"
        seen_inputs.add(key)
        c = _canon(mask)
        assert c not in seen_classes, "dihedral repeat of a motif within puzzle"
        seen_classes.add(c)
        # active-cell count: sum over present blocks of ink in their window
        expect = sum(int(mask[: i + 1, : j + 1].sum()) for i in range(h) for j in range(w) if mask[i, j])
        assert int((go != bg).sum()) == expect
    _check_discrimination(puzzle)


def self_test() -> None:
    shapes_out, bgs, regimes_shift, rect = set(), set(), 0, 0
    max_test_extrap = 0
    for seed in range(42):
        pz = generate_puzzle(seed)
        validate_generated_puzzle(pz)
        again = generate_puzzle(seed)
        assert all(np.array_equal(np.asarray(a["input"]), np.asarray(b["input"]))
                   for a, b in zip(pz["train"] + pz["test"], again["train"] + again["test"]))
        tr_max = max(max(np.asarray(p["input"]).shape) for p in pz["train"])
        te_max = max(max(np.asarray(p["input"]).shape) for p in pz["test"])
        if te_max > tr_max:
            regimes_shift += 1
            max_test_extrap = max(max_test_extrap, te_max)
        for p in pz["train"] + pz["test"]:
            gi = np.asarray(p["input"])
            shapes_out.add(np.asarray(p["output"]).shape)
            bgs.add(_roles(gi)[0])
            if gi.shape[0] != gi.shape[1]:
                rect += 1
    assert len(shapes_out) >= 10, shapes_out
    assert len(bgs) >= 6, bgs
    assert 8 <= regimes_shift <= 34, regimes_shift
    assert rect >= 20, rect
    assert max_test_extrap >= 6
    for a, b in ((2, 1), (10, 4), (5, 4), (3, 2)):
        validate_generated_puzzle(generate_puzzle(7, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (2, 2), (3, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test ok")
