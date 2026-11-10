"""Zoomed bracket-mosaic window.

The input is a magnified, camouflaged, possibly clipped window onto a small
concentric corner-bracket mosaic.  Complete one-cell background divider lines
partition the canvas into compartments (parent B); every compartment is either
empty or holds a stencil of one label colour and stands for exactly one logical
cell.  The resulting logical matrix is a window - always containing the mosaic
centre - onto a fourfold-symmetric assembly of concentric odd square layers made
of congruent corner brackets with an optional centre singleton (parent A).
The output is the complete mosaic at one cell per logical cell.
"""
from __future__ import annotations

import numpy as np

MAX_SIDE = 30
SIDES = ("up", "down", "left", "right")
STENCILS = ("solid", "dense", "medium", "sparse", "checker", "stripes", "walk")
STENCIL_P = (0.10, 0.27, 0.18, 0.12, 0.11, 0.10, 0.12)


# --------------------------------------------------------------------------
# Logical mosaic
# --------------------------------------------------------------------------

def _side_position(dy, dx, k):
    """Coordinate along the side of ring k for an offset on that ring."""
    return dx if abs(dy) == k else dy


def _bracket_on(t, k, arm):
    """Ring k (size 2k+1) with corner arm `arm` colours side position t."""
    return abs(t) >= k - arm + 1


def build_mosaic(n_layers, arms, layer_colors, center_color, bg):
    size = 2 * n_layers + 1
    m = np.full((size, size), bg, dtype=int)
    for r in range(size):
        for c in range(size):
            dy, dx = r - n_layers, c - n_layers
            k = max(abs(dy), abs(dx))
            if k == 0:
                if center_color is not None:
                    m[r, c] = center_color
            elif _bracket_on(_side_position(dy, dx, k), k, arms[k - 1]):
                m[r, c] = layer_colors[k - 1]
    return m


def _completions(window, bg):
    """All (centre, layers, arm options, singleton) configurations consistent
    with a logical window.  Each entry: (ci, cj, colors_by_ring, arm_options,
    singleton_color)."""
    rows, cols = window.shape
    colors = sorted(set(int(v) for v in window.flat) - {int(bg)})
    where = {c: np.argwhere(window == c) for c in colors}
    results = []
    for ci in range(rows):
        for cj in range(cols):
            radius = {}
            ok = True
            for c in colors:
                d = np.maximum(np.abs(where[c][:, 0] - ci), np.abs(where[c][:, 1] - cj))
                if d.min() != d.max():
                    ok = False
                    break
                radius[c] = int(d[0])
            if not ok:
                continue
            singleton = [c for c, r in radius.items() if r == 0]
            rings = {r: c for c, r in radius.items() if r > 0}
            if len(rings) != len([c for c, r in radius.items() if r > 0]):
                continue  # two colours on one ring
            n = len(rings)
            if n == 0 or set(rings) != set(range(1, n + 1)):
                continue
            reach = max(ci, rows - 1 - ci, cj, cols - 1 - cj)
            if reach > n:
                continue  # window must lie inside the mosaic
            arm_options = []
            for k in range(1, n + 1):
                cells = [(r - ci, c - cj) for r in range(rows) for c in range(cols)
                         if max(abs(r - ci), abs(c - cj)) == k]
                options = []
                for arm in range(1, k + 2):
                    if all((window[ci + dy, cj + dx] == rings[k]) ==
                           _bracket_on(_side_position(dy, dx, k), k, arm)
                           for dy, dx in cells):
                        options.append(arm)
                if not options:
                    ok = False
                    break
                arm_options.append(options)
            if not ok:
                continue
            results.append((ci, cj, [rings[k] for k in range(1, n + 1)], arm_options,
                            singleton[0] if singleton else None))
    return results


# --------------------------------------------------------------------------
# Physical decoding (parent B)
# --------------------------------------------------------------------------

def _runs(mask):
    """Maximal runs of False in a boolean divider mask -> list of (start, stop)."""
    runs, start = [], None
    for i, v in enumerate(list(mask) + [True]):
        if not v and start is None:
            start = i
        elif v and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _background(g):
    full_rows = {int(r[0]) for r in g if np.all(r == r[0])}
    full_cols = {int(c[0]) for c in g.T if np.all(c == c[0])}
    both = full_rows & full_cols
    pool = both if both else (full_rows | full_cols)
    if len(pool) != 1:
        raise ValueError("Cannot identify a unique divider/background colour")
    return pool.pop()


def decode(grid):
    """Return (background, logical window matrix, row bands, col bands)."""
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    row_bands = _runs(np.all(g == bg, axis=1))
    col_bands = _runs(np.all(g == bg, axis=0))
    m = np.full((len(row_bands), len(col_bands)), bg, dtype=int)
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            labels = set(int(v) for v in g[r0:r1, c0:c1].flat) - {bg}
            if len(labels) > 1:
                raise ValueError("Compartment holds more than one label colour")
            if labels:
                m[i, j] = labels.pop()
    return bg, m, row_bands, col_bands


def analyse(grid):
    """Full input-only analysis used by solve and validation."""
    bg, window, row_bands, col_bands = decode(grid)
    configs = _completions(window, bg)
    exact = [c for c in configs if all(len(o) == 1 for o in c[3])]
    return {"bg": bg, "window": window, "row_bands": row_bands, "col_bands": col_bands,
            "configs": configs, "exact": exact}


def solve(input_grid):
    info = analyse(input_grid)
    candidates = info["exact"] or info["configs"]
    if not candidates:
        raise ValueError("No consistent concentric completion")
    ci, cj, ring_colors, arm_options, singleton = candidates[0]
    arms = [opts[0] for opts in arm_options]
    return build_mosaic(len(ring_colors), arms, ring_colors, singleton, info["bg"])


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _band_sizes(rng, n):
    budget = MAX_SIDE - (n - 1)
    style = rng.choice(["free", "even", "contrast", "small"], p=[0.45, 0.15, 0.25, 0.15])
    weights = np.array([1, 3, 4, 4, 3, 2, 1], dtype=float)
    if style == "even":
        sizes = [int(rng.choice(np.arange(1, 8), p=weights / weights.sum()))] * n
    elif style == "contrast":
        sizes = [int(rng.integers(1, 4)) for _ in range(n)]
        sizes[int(rng.integers(0, n))] = int(rng.integers(5, 11))
    elif style == "small":
        sizes = [int(rng.integers(1, 3)) for _ in range(n)]
    else:
        sizes = [int(rng.choice(np.arange(1, 8), p=weights / weights.sum())) for _ in range(n)]
    while sum(sizes) > budget:
        sizes[int(np.argmax(sizes))] -= 1
    return sizes


def _stencil(rng, h, w):
    style = STENCILS[int(rng.choice(len(STENCILS), p=STENCIL_P))]
    if style == "solid":
        mask = np.ones((h, w), dtype=bool)
    elif style == "dense":
        mask = rng.random((h, w)) < rng.uniform(0.7, 0.9)
    elif style == "medium":
        mask = rng.random((h, w)) < rng.uniform(0.45, 0.65)
    elif style == "sparse":
        mask = rng.random((h, w)) < rng.uniform(0.15, 0.35)
    elif style == "checker":
        ii, jj = np.indices((h, w))
        mask = (ii + jj) % 2 == int(rng.integers(0, 2))
    elif style == "stripes":
        ii, jj = np.indices((h, w))
        mask = (ii if rng.random() < 0.5 else jj) % 2 == int(rng.integers(0, 2))
    else:  # random walk
        mask = np.zeros((h, w), dtype=bool)
        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
        for _ in range(max(1, int(h * w * rng.uniform(0.6, 1.4)))):
            mask[r, c] = True
            r = int(np.clip(r + rng.integers(-1, 2), 0, h - 1))
            c = int(np.clip(c + rng.integers(-1, 2), 0, w - 1))
    # Every compartment row and column carries label, so the only complete
    # background lines are the dividers.
    for r in range(h):
        if not mask[r].any():
            mask[r, int(rng.integers(0, w))] = True
    for c in range(w):
        if not mask[:, c].any():
            mask[int(rng.integers(0, h)), c] = True
    return mask


def render(rng, window, bg):
    rows, cols = window.shape
    hs, ws = _band_sizes(rng, rows), _band_sizes(rng, cols)
    grid = np.full((sum(hs) + rows - 1, sum(ws) + cols - 1), bg, dtype=int)
    r0 = 0
    for i, h in enumerate(hs):
        c0 = 0
        for j, w in enumerate(ws):
            if window[i, j] != bg:
                grid[r0:r0 + h, c0:c0 + w][_stencil(rng, h, w)] = window[i, j]
            c0 += w + 1
        r0 += h + 1
    return grid


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------

def _window_extents(rng, n, role):
    if role == "full":
        return {s: n for s in SIDES}
    if role == "clip1":
        clipped = [SIDES[int(rng.integers(0, 4))]]
    else:
        count = 2 if rng.random() < 0.75 else 3
        clipped = list(rng.choice(SIDES, size=count, replace=False))
    ext = {}
    for s in SIDES:
        if s in clipped:
            ext[s] = int(rng.integers(0, n)) if rng.random() < 0.6 else int(rng.integers(max(0, n - 2), n))
        else:
            ext[s] = n
    return ext


def _clipped_sides(info_config, window_shape):
    ci, cj, rings = info_config[0], info_config[1], info_config[2]
    n = len(rings)
    rows, cols = window_shape
    ext = {"up": ci, "down": rows - 1 - ci, "left": cj, "right": cols - 1 - cj}
    return [s for s in SIDES if ext[s] < n]


def _make_pair(rng, role, with_center, n_choices, bg_hint):
    for _ in range(400):
        n = int(rng.choice(n_choices))
        if bg_hint is not None and rng.random() < 0.6:
            bg = bg_hint
            others = [c for c in rng.permutation(10).tolist() if c != bg]
        else:
            perm = rng.permutation(10).tolist()
            bg, others = perm[0], perm[1:]
        layer_colors = others[:n]
        center_color = others[n] if with_center else None
        arms = []
        for k in range(1, n + 1):
            if rng.random() < 0.25:
                arms.append(k + 1)  # full outline
            else:
                arms.append(int(rng.integers(1, k + 2)))
        mosaic = build_mosaic(n, arms, layer_colors, center_color, bg)
        ext = _window_extents(rng, n, role)
        window = mosaic[n - ext["up"]:n + ext["down"] + 1, n - ext["left"]:n + ext["right"] + 1]
        rows, cols = window.shape
        if rows < 2 or cols < 2 or rows > 11 or cols > 11:
            continue
        if not all((window != bg).any(axis=1)) or not all((window != bg).any(axis=0)):
            continue
        if set(layer_colors) - set(int(v) for v in window.flat):
            continue
        configs = _completions(window, bg)
        if len(configs) != 1 or any(len(o) != 1 for o in configs[0][3]):
            continue
        grid = render(rng, window, bg)
        if not np.array_equal(solve(grid), mosaic):
            continue
        meta = {"n": n, "arms": arms, "center": with_center, "role": role,
                "clipped": [s for s in SIDES if ext[s] < n]}
        return {"input": grid, "output": mosaic}, meta
    raise RuntimeError("Could not build a pair")


def _check_counts(num_train, num_test):
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("Need 2..10 training pairs, 1..4 test pairs and num_train > num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    num_train, num_test = int(num_train), int(num_test)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        train_roles = ["full", "clip1"] + [str(rng.choice(["full", "clip1", "clip1", "clip2"]))
                                           for _ in range(num_train - 2)]
        rng.shuffle(train_roles)
        centers = [True, False] + [bool(rng.random() < 0.5) for _ in range(num_train - 2)]
        rng.shuffle(centers)
        test_roles = ["clip2"] + [str(rng.choice(["full", "clip1", "clip2", "clip2"]))
                                  for _ in range(num_test - 1)]
        test_centers = [bool(rng.random() < 0.5) for _ in range(num_test)]
        bg_hint = int(rng.integers(0, 10)) if rng.random() < 0.5 else None
        pairs, metas, seen = [], [], set()
        specs = list(zip(train_roles, centers)) + list(zip(test_roles, test_centers))
        for index, (role, center) in enumerate(specs):
            is_test = index >= num_train
            if role == "full":
                n_choices = [1, 2, 2, 3, 3, 3, 4, 4]
            elif is_test and index == num_train:
                n_choices = [2, 3, 3, 4, 4, 4, 5, 5, 6]
            else:
                n_choices = [2, 2, 3, 3, 4, 4, 5]
            for _ in range(20):
                pair, meta = _make_pair(rng, role, center, n_choices, bg_hint)
                key = pair["input"].tobytes() + bytes(pair["input"].shape)
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Could not make distinct inputs")
            pairs.append(pair)
            metas.append(meta)
        puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("Could not build a valid puzzle")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _describe(pair):
    info = analyse(pair["input"])
    assert len(info["configs"]) == 1, "Completion is not unique"
    config = info["configs"][0]
    assert all(len(o) == 1 for o in config[3]), "Arm length is ambiguous"
    window = info["window"]
    n = len(config[2])
    runs = False
    for k in range(n):
        color = config[2][k]
        same = window == color
        if (same[:, 1:] & same[:, :-1]).any() or (same[1:, :] & same[:-1, :]).any():
            runs = True
    arms = [o[0] for o in config[3]]
    return {
        "bg": info["bg"], "n": n, "arms": arms, "center": config[4] is not None,
        "clipped": _clipped_sides(config, window.shape), "window": window,
        "adjacent_repeat": runs, "gap": any(a < k + 1 for k, a in enumerate(arms, 1)),
        "row_bands": [b - a for a, b in info["row_bands"]],
        "col_bands": [b - a for a, b in info["col_bands"]],
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    seen = set()
    descriptions = {}
    for split in ("train", "test"):
        descriptions[split] = []
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            gi = np.asarray(pair["input"], dtype=int)
            go = np.asarray(pair["output"], dtype=int)
            for g in (gi, go):
                assert g.ndim == 2 and 1 <= g.shape[0] <= MAX_SIDE and 1 <= g.shape[1] <= MAX_SIDE
                assert g.min() >= 0 and g.max() <= 9
            key = gi.tobytes() + bytes(gi.shape)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            assert np.array_equal(solve(gi), go), "solve disagrees with stored output"
            d = _describe({"input": gi, "output": go})
            assert len(d["row_bands"]) >= 2 and len(d["col_bands"]) >= 2
            assert go.shape == (2 * d["n"] + 1,) * 2
            descriptions[split].append(d)
    train, test = descriptions["train"], descriptions["test"]
    assert any(not d["clipped"] for d in train), "Training needs a fully visible mosaic"
    assert any(d["clipped"] for d in train), "Training needs a clipped window"
    assert {d["center"] for d in train} == {True, False}, "Training needs both centre states"
    assert len(test[0]["clipped"]) >= 2, "First test must be clipped on at least two sides"
    assert any(d["adjacent_repeat"] for d in train), "Training must show repeated adjacent labels"
    assert any(d["gap"] for d in train), "Training must show bracket gaps"
    # Compress-only shortcut (parent B alone) fails on a training pair and the first test.
    for group in (train, test[:1]):
        assert any(d["window"].shape != (2 * d["n"] + 1,) * 2 for d in group)
    return True


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def self_test():
    stats = {"n": set(), "clip_counts": set(), "bg": set(), "center_test": set(),
             "shapes": set(), "band_sizes": set(), "test_n": set(), "clip_sides": set(),
             "windows_test_asym": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        for split in ("train", "test"):
            for a, b in zip(puzzle[split], again[split]):
                assert np.array_equal(a["input"], b["input"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _describe(pair)
                stats["n"].add(d["n"])
                stats["clip_counts"].add(len(d["clipped"]))
                stats["clip_sides"].update(d["clipped"])
                stats["bg"].add(d["bg"])
                stats["shapes"].add(np.asarray(pair["input"]).shape)
                stats["band_sizes"].update(d["row_bands"] + d["col_bands"])
                if split == "test":
                    stats["center_test"].add(d["center"])
                    stats["test_n"].add(d["n"])
    assert stats["n"] >= {1, 2, 3, 4, 5}, stats["n"]
    assert stats["clip_counts"] >= {0, 1, 2, 3}, stats["clip_counts"]
    assert stats["clip_sides"] == set(SIDES)
    assert len(stats["bg"]) >= 8
    assert stats["center_test"] == {True, False}
    assert len(stats["shapes"]) >= 60
    assert min(stats["band_sizes"]) == 1 and max(stats["band_sizes"]) >= 8
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (3, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
