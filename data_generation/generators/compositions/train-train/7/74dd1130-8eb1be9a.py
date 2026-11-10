"""Diagonal-mirror stripe folding.

Composition of parent A (74dd1130, main-diagonal transpose) and parent B
(8eb1be9a, phase-anchored periodic extension of a complete row band).

Input: a uniform background, one complete band of k (2..4) consecutive full
rows (or full columns) that defines a periodic stripe field, and one or two
marker-coloured mirror segments parallel to the main diagonal.  Each segment
runs from one edge to an adjacent edge and cuts off a corner triangle (the
top-right or bottom-left corner) that does not contain the band.

Output: the band is extended periodically over the whole canvas with its
original phase (parent B).  The mirror cells keep their colour.  Every cell of
a cut-off triangle shows the reflection of the completed stripe field across
its mirror segment, i.e. the shifted transpose (r, c) -> (c + d, r - d) for the
segment r - c = d (parent A).  Horizontal stripes therefore bend into vertical
stripes at the mirror (and vice versa).
"""
from __future__ import annotations

import numpy as np

MIN_LINE = 4          # shortest mirror segment
MAX_K = 4             # thickest band
MAX_SIDE = 30
CORNERS = ("TR", "BL")

_RELATIONS = {
    2: (((0, 1), 0.86), ((0, 0), 0.14)),
    3: (((0, 1, 2), 0.38), ((0, 1, 0), 0.2), ((0, 1, 1), 0.18),
        ((0, 0, 1), 0.18), ((0, 0, 0), 0.06)),
    4: (((0, 1, 2, 3), 0.30), ((0, 1, 0, 2), 0.14), ((0, 1, 1, 0), 0.12),
        ((0, 0, 1, 1), 0.12), ((0, 1, 2, 1), 0.12), ((0, 0, 0, 1), 0.10),
        ((0, 1, 2, 0), 0.10)),
}
_MASK_KINDS = (("solid", 0.22), ("sparse", 0.2), ("dense", 0.1), ("half", 0.14),
               ("shift", 0.14), ("complement", 0.1), ("random", 0.1))
_TIERS = {
    # canonical width range, gap range above/below the band
    "small": ((5, 8), (0, 3)),
    "medium": ((7, 11), (0, 5)),
    "large": ((10, 15), (1, 7)),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Input-only analysis shared by solve() and the validators
# ---------------------------------------------------------------------------

def _segments(points, height, width):
    """Corner-cutting main-direction segments formed by a colour, or None."""
    diagonals = {}
    for r, c in points:
        diagonals.setdefault(int(r) - int(c), []).append(int(r))
    if not 1 <= len(diagonals) <= 2:
        return None
    found = []
    for d, rows in diagonals.items():
        rows = sorted(rows)
        if rows[-1] - rows[0] + 1 != len(rows) or len(rows) < MIN_LINE:
            return None
        r0, r1 = rows[0], rows[-1]
        c0, c1 = r0 - d, r1 - d
        top_right = r0 == 0 and c1 == width - 1
        bottom_left = c0 == 0 and r1 == height - 1
        if top_right == bottom_left:
            return None
        found.append((d, "TR" if top_right else "BL"))
    if len(found) == 2 and found[0][1] == found[1][1]:
        return None
    return tuple(sorted(found, key=lambda item: item[1], reverse=True))


def _analyse(grid):
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("expected a 2D grid")
    g = g.astype(int)
    height, width = g.shape
    values, counts = np.unique(g, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(values) < 3 or counts[order[0]] == counts[order[1]]:
        raise ValueError("no unique background")
    background = int(values[order[0]])
    markers = []
    for value in values:
        value = int(value)
        if value == background:
            continue
        segs = _segments(np.argwhere(g == value), height, width)
        if segs is not None:
            markers.append((value, segs))
    if len(markers) != 1:
        raise ValueError("mirror colour is not unique")
    marker, segs = markers[0]
    band_mask = (g != background) & (g != marker)
    rows = np.flatnonzero(band_mask.any(axis=1))
    cols = np.flatnonzero(band_mask.any(axis=0))

    def contiguous(index):
        return len(index) > 0 and index[-1] - index[0] + 1 == len(index)

    if contiguous(rows) and len(rows) <= MAX_K and len(cols) > MAX_K:
        orientation, start, k = "h", int(rows[0]), len(rows)
    elif contiguous(cols) and len(cols) <= MAX_K and len(rows) > MAX_K:
        orientation, start, k = "v", int(cols[0]), len(cols)
    else:
        raise ValueError("no unique band")
    return {"grid": g, "height": height, "width": width, "background": background,
            "marker": marker, "segments": segs, "orientation": orientation,
            "start": start, "k": k}


def _field(info, anchor=None):
    """Periodic extension of the band over the canvas (parent B)."""
    g, start, k = info["grid"], info["start"], info["k"]
    anchor = start if anchor is None else anchor
    if info["orientation"] == "h":
        index = start + (np.arange(info["height"]) - anchor) % k
        return g[index, :].copy()
    index = start + (np.arange(info["width"]) - anchor) % k
    return g[:, index].copy()


def _targets(info, d, corner):
    rr, cc = np.indices((info["height"], info["width"]))
    mask = (rr - cc < d) if corner == "TR" else (rr - cc > d)
    return np.nonzero(mask)


def _render(info, mode="rule"):
    """Rule output, or a named shortcut used to check the demonstrations."""
    g, background, marker = info["grid"], info["background"], info["marker"]
    height, width = info["height"], info["width"]
    if mode == "phase_start":
        field = _field(info, anchor=0)
    elif mode == "phase_end":
        size = info["height"] if info["orientation"] == "h" else info["width"]
        field = _field(info, anchor=size - info["k"])
    elif mode == "a_only":
        field = g.copy()
    else:
        field = _field(info)
    out = field.copy()
    for d, corner in info["segments"]:
        tr, tc = _targets(info, d, corner)
        if mode in ("rule", "phase_start", "phase_end", "a_only"):
            sr, sc = tc + d, tr - d
            if (sr.min(initial=0) < 0 or sc.min(initial=0) < 0
                    or sr.max(initial=0) >= height or sc.max(initial=0) >= width):
                raise ValueError("mirror image leaves the canvas")
            out[tr, tc] = field[sr, sc]
        elif mode == "blank":
            out[tr, tc] = background
        elif mode == "plain_transpose":
            # Ignore the mirror offset: use the unshifted transpose of the field.
            for r, c in zip(tr, tc):
                if info["orientation"] == "h" and r < width:
                    out[r, c] = g[info["start"] + (c - info["start"]) % info["k"], r]
                elif info["orientation"] == "v" and c < height:
                    out[r, c] = g[c, info["start"] + (r - info["start"]) % info["k"]]
                else:
                    out[r, c] = -1
        elif mode == "b_only":
            pass
        else:
            raise ValueError(mode)
    out[g == marker] = marker
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Extend the band periodically, then fold the field across each mirror."""
    return _render(_analyse(input_grid), "rule").astype(int)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _count(value, name, lower, upper, default):
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _weighted(rng, options):
    values = [item[0] for item in options]
    weights = np.asarray([item[1] for item in options], dtype=float)
    return values[int(rng.choice(len(values), p=weights / weights.sum()))]


def _mask(rng, kind, length, previous):
    if kind == "solid":
        mask = np.ones(length, dtype=bool)
    elif kind == "sparse":
        mask = np.zeros(length, dtype=bool)
        mask[int(rng.integers(length))] = True
    elif kind == "dense":
        mask = np.ones(length, dtype=bool)
        mask[int(rng.integers(length))] = False
    elif kind == "half":
        mask = np.zeros(length, dtype=bool)
        mask[rng.choice(length, size=(length + 1) // 2, replace=False)] = True
    elif kind == "shift" and previous is not None:
        mask = np.roll(previous, int(rng.integers(1, length)))
    elif kind == "complement" and previous is not None and 0 < previous.sum() < length:
        mask = ~previous
    else:
        mask = rng.random(length) < float(rng.uniform(0.3, 0.7))
    if not mask.any():
        mask[int(rng.integers(length))] = True
    return mask


def _make_band(rng, k, width, background, colors):
    """A k-row band: B-style rhythmic or irregular rows with a row relation."""
    word = _weighted(rng, _RELATIONS[k])
    letters = max(word) + 1
    rhythmic = rng.random() < 0.76
    per_cell = len(colors) > 1 and rng.random() < 0.45
    for _ in range(80):
        rows, previous = [], None
        if rhythmic:
            period = int(rng.integers(2, min(6, width - 1) + 1))
            offset = int(rng.integers(period))
            phase = (np.arange(width) + offset) % period
            colour_tile = rng.choice(colors, size=period)
        for letter in range(letters):
            if rhythmic:
                kind = _weighted(rng, _MASK_KINDS)
                tile = _mask(rng, kind, period, previous)
                previous = tile
                fg = tile[phase]
                if per_cell:
                    paint = colour_tile[phase]
                else:
                    paint = np.full(width, int(rng.choice(colors)))
            else:
                if rng.random() < 0.3:
                    fg = np.zeros(width, dtype=bool)
                    for _ in range(int(rng.integers(1, 4))):
                        left = int(rng.integers(width))
                        fg[left:left + int(rng.integers(1, max(2, width // 2)))] = True
                else:
                    fg = rng.random(width) < float(rng.uniform(0.2, 0.6))
                if not fg.any():
                    fg[int(rng.integers(width))] = True
                if per_cell:
                    paint = rng.choice(colors, size=width)
                else:
                    paint = np.full(width, int(rng.choice(colors)))
            row = np.where(fg, paint, background).astype(int)
            rows.append(row)
        keys = {row.tobytes() for row in rows}
        if len(keys) != letters:
            continue
        band = np.stack([rows[i] for i in word])
        if (band != background).any(axis=0).sum() <= MAX_K:
            continue
        return band
    raise RuntimeError("could not build a band")


def _palette(rng, palette):
    """Pair colours.  Mirror colours of the puzzle never appear inside a band."""
    markers = palette["markers"]
    marker = int(markers[0] if len(markers) == 1 else rng.choice(markers))
    if palette["mode"] == "fixed":
        background = palette["background"]
    else:
        free = [c for c in range(10) if c not in markers]
        background = 0 if 0 in free and rng.random() < 0.35 else int(rng.choice(free))
    others = [c for c in range(10) if c not in markers and c != background]
    count = int(rng.choice((1, 2, 3), p=(0.55, 0.3, 0.15)))
    colors = [int(c) for c in rng.choice(others, size=count, replace=False)]
    return background, marker, colors


def _canonical_corners(spec):
    if spec["orientation"] == "h":
        return set(spec["corners"])
    return {"BL" if c == "TR" else "TR" for c in spec["corners"]}


def _build_pair(rng, spec, palette):
    """Build in the canonical frame (horizontal band) and transpose if needed."""
    k = spec["k"]
    corners = _canonical_corners(spec)
    (w_lo, w_hi), (g_lo, g_hi) = _TIERS[spec["tier"]]
    for _ in range(400):
        width = int(rng.integers(w_lo, w_hi + 1))
        top = int(rng.integers(MIN_LINE, width + 1)) if "TR" in corners else 0
        bottom = int(rng.integers(MIN_LINE, width + 1)) if "BL" in corners else 0
        above = int(rng.integers(g_lo, g_hi + 1))
        below = int(rng.integers(g_lo, g_hi + 1))
        if "TR" not in corners and rng.random() < 0.2:
            above = 0                      # band touches the top edge
        if "BL" not in corners and rng.random() < 0.2:
            below = 0                      # band touches the bottom edge
        height = top + above + k + below + bottom
        if height < k + 2 * k or height > MAX_SIDE:
            continue
        background, marker, colors = _palette(rng, palette)
        band = _make_band(rng, k, width, background, colors)
        start = top + above
        grid = np.full((height, width), background, dtype=int)
        grid[start:start + k] = band
        if "TR" in corners:
            e = width - top
            for i in range(top):
                grid[i, e + i] = marker
        if "BL" in corners:
            d = height - bottom
            for i in range(bottom):
                grid[d + i, i] = marker
        output = _construct(grid, band, start, k, corners, width - top, height - bottom, marker)
        if spec["orientation"] == "v":
            grid, output = grid.T.copy(), output.T.copy()
        try:
            info = _analyse(grid)
        except ValueError:
            continue
        if (info["orientation"] != spec["orientation"] or info["k"] != k
                or tuple(c for _, c in info["segments"]) != tuple(sorted(spec["corners"], reverse=True))):
            continue
        if not _pair_is_informative(info, output):
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("could not build a pair")


def _construct(grid, band, start, k, corners, e, d_bottom, marker):
    """Forward construction in the canonical frame with explicit coordinates."""
    height, width = grid.shape
    field = np.stack([band[(r - start) % k] for r in range(height)])
    out = field.copy()
    for r in range(height):
        for c in range(width):
            if "TR" in corners and c - r > e:          # above segment r - c = -e
                out[r, c] = field[c - e, r + e]
            if "BL" in corners and r - c > d_bottom:   # below segment r - c = d
                out[r, c] = field[c + d_bottom, r - d_bottom]
    out[grid == marker] = marker
    return out


def _pair_is_informative(info, output):
    """Both parent concepts must change this pair's answer."""
    if not np.array_equal(_render(info, "rule"), output):
        return False
    background = info["background"]
    for d, corner in info["segments"]:
        tr, tc = _targets(info, d, corner)
        if np.count_nonzero(output[tr, tc] != background) < 2:
            return False
    for mode in ("b_only", "blank", "a_only"):
        if np.array_equal(_render(info, mode), output):
            return False
    # The fold must be visible: a quarter of the triangle cells differ from
    # simply continuing the stripes straight through the mirror.
    unfolded = _render(info, "b_only")
    for d, corner in info["segments"]:
        tr, tc = _targets(info, d, corner)
        changed = np.count_nonzero(output[tr, tc] != unfolded[tr, tc])
        if changed < max(2, int(np.ceil(0.25 * len(tr)))):
            return False
    # The completed field must be visibly larger than the supplied band.
    return np.count_nonzero(output != info["grid"]) >= 6


def _plan(rng, n_train, n_test):
    k_mode = "fixed3" if rng.random() < 0.55 else "variable"
    palette_mode = _weighted(rng, (("fixed", 0.55), ("background_varies", 0.25), ("all_vary", 0.2)))
    shifts = [("scale", 0.3), ("double", 0.3), ("orientation", 0.25)]
    if k_mode == "variable":
        shifts.append(("thickness", 0.3))
    shift = _weighted(rng, shifts)
    if shift == "orientation":
        orientation_mode = "mixed"
    else:
        orientation_mode = _weighted(rng, (("h", 0.33), ("v", 0.27), ("mixed", 0.4)))
    major = "h" if rng.random() < 0.55 else "v"
    minor = "v" if major == "h" else "h"
    double_mix = shift != "double" and rng.random() < 0.3

    # orientations
    if orientation_mode in ("h", "v"):
        train_o = [orientation_mode] * n_train
        test_o = [orientation_mode] * n_test
    elif shift == "orientation":
        count = 1 if n_train <= 4 else int(rng.integers(1, 3))
        train_o = [minor] * count + [major] * (n_train - count)
        test_o = [minor] * n_test
    else:
        train_o = [major, minor] + [str(rng.choice(("h", "v"))) for _ in range(n_train - 2)]
        test_o = [str(rng.choice(("h", "v"))) for _ in range(n_test)]

    # mirror corners
    def single():
        return (str(rng.choice(CORNERS)),)

    if shift == "double":
        count = 1 if n_train <= 4 else int(rng.integers(1, 3))
        train_c = [("TR", "BL")] * count + [single() for _ in range(n_train - count)]
        test_c = [("TR", "BL")] * n_test
    elif double_mix:
        train_c = [("TR", "BL")] + [("TR", "BL") if rng.random() < 0.3 else single()
                                    for _ in range(n_train - 1)]
        test_c = [("TR", "BL") if rng.random() < 0.35 else single() for _ in range(n_test)]
    else:
        train_c = [("TR",), ("BL",)] + [single() for _ in range(n_train - 2)]
        test_c = [single() for _ in range(n_test)]

    # band thickness
    if k_mode == "fixed3":
        train_k, test_k = [3] * n_train, [3] * n_test
    elif shift == "thickness":
        common, rare = (int(v) for v in rng.choice((2, 3, 4), size=2, replace=False))
        train_k = [rare] + [common] * (n_train - 1)
        test_k = [rare] * n_test
    else:
        first, second = (int(v) for v in rng.choice((2, 3, 4), size=2, replace=False))
        train_k = [first, second] + [int(rng.choice((2, 3, 4), p=(0.3, 0.4, 0.3)))
                                     for _ in range(n_train - 2)]
        test_k = [int(rng.choice(train_k)) for _ in range(n_test)]

    # canvas scale
    if shift == "scale":
        train_t = [str(rng.choice(("small", "medium"), p=(0.5, 0.5))) for _ in range(n_train)]
        test_t = ["large"] * n_test
    else:
        tiers = ("small", "medium", "large")
        train_t = [str(rng.choice(tiers, p=(0.38, 0.42, 0.2))) for _ in range(n_train)]
        test_t = [str(rng.choice(tiers, p=(0.3, 0.45, 0.25))) for _ in range(n_test)]

    order = rng.permutation(n_train)
    train = [{"orientation": train_o[i], "corners": train_c[i], "k": train_k[i],
              "tier": train_t[i]} for i in order]
    # shuffle each attribute list independently so features are not tied together
    for key, values in (("corners", train_c), ("k", train_k), ("tier", train_t)):
        perm = rng.permutation(n_train)
        for index, spec in enumerate(train):
            spec[key] = values[perm[index]]
    test = [{"orientation": test_o[i], "corners": test_c[i], "k": test_k[i],
             "tier": test_t[i]} for i in range(n_test)]
    background = 0 if rng.random() < 0.6 else int(rng.integers(10))
    marker_count = 2 if palette_mode == "all_vary" else 1
    markers = [int(c) for c in rng.choice([c for c in range(10) if c != background],
                                          size=marker_count, replace=False)]
    palette = {"mode": palette_mode, "background": background, "markers": markers}
    return {"shift": shift, "k_mode": k_mode, "palette": palette, "train": train, "test": test}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n_train = _count(num_train, "num_train", 2, 10, 4)
    n_test = _count(num_test, "num_test", 1, 4, 1)
    if n_train <= n_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    for _ in range(60):
        plan = _plan(rng, n_train, n_test)
        puzzle = {"train": [], "test": []}
        seen = set()
        try:
            for split in ("train", "test"):
                for spec in plan[split]:
                    for _ in range(30):
                        pair = _build_pair(rng, spec, plan["palette"])
                        key = (pair["input"].shape, pair["input"].tobytes())
                        if key not in seen:
                            break
                    else:
                        raise RuntimeError("duplicate inputs")
                    seen.add(key)
                    puzzle[split].append(pair)
            if plan["shift"] == "scale":
                biggest = max(p["input"].size for p in puzzle["train"])
                if min(p["input"].size for p in puzzle["test"]) <= biggest:
                    continue
            validate_generated_puzzle(puzzle)
        except (RuntimeError, AssertionError, ValueError):
            continue
        return puzzle
    raise RuntimeError("unable to generate a valid puzzle")


# ---------------------------------------------------------------------------
# Validation and self-test
# ---------------------------------------------------------------------------

def _as_grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= MAX_SIDE for n in array.shape), "bad grid shape")
    _require(np.issubdtype(array.dtype, np.integer), "grid cells must be integers")
    _require(np.all((array >= 0) & (array <= 9)), "grid colours must be 0..9")
    return array.astype(int)


def _check_rule_properties(info, output):
    """Property checks written independently of _render."""
    g, k, start = info["grid"], info["k"], info["start"]
    height, width = g.shape
    marker, background = info["marker"], info["background"]
    _require(output.shape == g.shape, "output shape must equal input shape")
    # the input contains only background, band and mirror cells
    band_slice = (slice(start, start + k), slice(None)) if info["orientation"] == "h" \
        else (slice(None), slice(start, start + k))
    outside = g.copy()
    outside[band_slice] = background
    outside[g == marker] = background
    _require(np.all(outside == background), "stray cells outside band and mirrors")
    _require(not np.any(g[band_slice] == marker), "mirror crosses the band")
    _require(np.all((g[band_slice] != background).any(axis=1 if info["orientation"] == "h" else 0)),
             "every band line must contain foreground")
    _require(np.all(output[g == marker] == marker), "mirror cells must persist")
    _require(np.array_equal(output[band_slice], g[band_slice]), "band must be preserved")
    target = np.zeros(g.shape, dtype=bool)
    for d, corner in info["segments"]:
        tr, tc = _targets(info, d, corner)
        _require(not np.any(target[tr, tc]), "mirror triangles overlap")
        target[tr, tc] = True
        _require(np.all(g[tr, tc] == background), "triangle must be blank in the input")
    source = ~target & (g != marker)
    for d, corner in info["segments"]:
        tr, tc = _targets(info, d, corner)
        sr, sc = tc + d, tr - d
        _require(np.all(source[sr, sc]), "mirror image must lie in the stripe field")
        _require(np.array_equal(output[tr, tc], output[sr, sc]), "triangle is not a reflection")
    # periodicity of the stripe field with the band's own period and phase
    rr, cc = np.nonzero(source)
    for r, c in zip(rr, cc):
        if info["orientation"] == "h":
            expected = g[start + (r - start) % k, c]
        else:
            expected = g[r, start + (c - start) % k]
        _require(output[r, c] == expected, "stripe field is not the anchored band period")


def _features(info):
    return {"orientation": info["orientation"], "k": info["k"],
            "corners": tuple(c for _, c in info["segments"]),
            "offsets": tuple(d for d, _ in info["segments"])}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            grid, output = _as_grid(pair["input"]), _as_grid(pair["output"])
            key = (grid.shape, grid.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyse(grid)
            _check_rule_properties(info, output)
            _require(np.array_equal(solve(grid), output), "solve disagrees with output")
            _require(_pair_is_informative(info, output), "pair does not need both concepts")
            infos[split].append(info)
    every = infos["train"] + infos["test"]
    mirror_colours = {i["marker"] for i in every}
    band_colours = set()
    for i in every:
        band_colours |= set(np.unique(i["grid"]).tolist()) - {i["background"], i["marker"]}
    _require(not mirror_colours & band_colours, "a mirror colour also appears inside a band")
    _require(not mirror_colours & {i["background"] for i in every},
             "a mirror colour is also a background")
    train_f = [_features(i) for i in infos["train"]]
    test_f = [_features(i) for i in infos["test"]]
    # every test feature is demonstrated in training
    _require({f["orientation"] for f in test_f} <= {f["orientation"] for f in train_f},
             "test band orientation never demonstrated")
    _require({c for f in test_f for c in f["corners"]} <= {c for f in train_f for c in f["corners"]},
             "test mirror corner never demonstrated")
    if any(len(f["corners"]) == 2 for f in test_f):
        _require(any(len(f["corners"]) == 2 for f in train_f), "double mirror never demonstrated")
    train_k = {f["k"] for f in train_f}
    _require({f["k"] for f in test_f} <= train_k, "test band thickness never demonstrated")
    if len({f["k"] for f in train_f + test_f}) > 1:
        _require(len(train_k) >= 2, "variable thickness needs two training thicknesses")
    # at least one clearly visible fold (mirror segment of length >= 5)
    _require(any(len(_targets(info, d, c)[0]) >= 10
                 for info in infos["train"] for d, c in info["segments"]),
             "training lacks a sizeable mirror triangle")
    # training refutes the plausible shortcuts
    for mode in ("plain_transpose", "phase_start", "phase_end"):
        _require(any(not np.array_equal(_render(info, mode), _as_grid(pair["output"]))
                     for info, pair in zip(infos["train"], train)),
                 f"training does not refute shortcut {mode}")


def _signature(puzzle):
    return tuple((split, tuple((np.asarray(p["input"]).tobytes(), np.asarray(p["output"]).tobytes(),
                                np.asarray(p["input"]).shape) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test() -> None:
    for args in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(5, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    for seed, a, b in ((11, 2, 1), (12, 10, 4), (13, 5, 4), (14, 3, 2)):
        puzzle = generate_puzzle(seed, a, b)
        _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "counts ignored")
        validate_generated_puzzle(puzzle)
    _require(_signature(generate_puzzle(77)) == _signature(generate_puzzle(77)), "nondeterministic")

    signatures, shapes = set(), set()
    orientations, corner_sets, ks, offsets = set(), set(), set(), set()
    backgrounds, markers, band_colours = set(), set(), set()
    mixed_orientation = double_tests = varied_k = bigger_tests = zero_offset = 0
    edge_bands = interior_bands = multicolour = background_varies = pure_transposes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        infos = {s: [_analyse(p["input"]) for p in puzzle[s]] for s in ("train", "test")}
        train_o = {i["orientation"] for i in infos["train"]}
        mixed_orientation += len(train_o) == 2
        double_tests += any(len(i["segments"]) == 2 for i in infos["test"])
        varied_k += len({i["k"] for i in infos["train"]}) > 1
        bigger_tests += (min(i["grid"].size for i in infos["test"])
                         > max(i["grid"].size for i in infos["train"]))
        background_varies += len({i["background"] for i in infos["train"] + infos["test"]}) > 1
        for pair, info in zip(puzzle["train"] + puzzle["test"], infos["train"] + infos["test"]):
            output = np.asarray(pair["output"])
            for d, corner in info["segments"]:
                if d == 0:   # parent A literally: the triangle is the matrix transpose
                    tr, tc = _targets(info, d, corner)
                    _require(np.array_equal(output[tr, tc], output.T[tr, tc]), "d=0 fold is not a transpose")
                    pure_transposes += 1
            shapes.add(info["grid"].shape)
            orientations.add(info["orientation"])
            corner_sets.add(tuple(c for _, c in info["segments"]))
            ks.add(info["k"])
            for d, _ in info["segments"]:
                offsets.add(d)
                zero_offset += d == 0
            backgrounds.add(info["background"])
            markers.add(info["marker"])
            colours = set(np.unique(info["grid"]).tolist()) - {info["background"], info["marker"]}
            band_colours |= colours
            multicolour += len(colours) > 1
            size = info["height"] if info["orientation"] == "h" else info["width"]
            if info["start"] in (0, size - info["k"]):
                edge_bands += 1
            else:
                interior_bands += 1
    _require(len(signatures) == 42, "seeds must give distinct puzzles")
    _require(len(shapes) >= 60, "shape diversity")
    _require(orientations == {"h", "v"}, "both band orientations")
    _require({("TR",), ("BL",), ("TR", "BL")} <= corner_sets, "mirror configurations")
    _require(ks == {2, 3, 4}, "band thickness diversity")
    _require(len(offsets) >= 15 and zero_offset >= 2, "mirror offset diversity")
    _require(len(backgrounds) >= 6 and len(markers) >= 8 and len(band_colours) >= 9,
             "palette diversity")
    _require(mixed_orientation >= 8 and double_tests >= 6 and varied_k >= 6 and bigger_tests >= 6,
             "train/test shift diversity")
    _require(edge_bands >= 10 and interior_bands >= 60 and multicolour >= 20,
             "band placement and colour diversity")
    _require(background_varies >= 5 and pure_transposes >= 2, "palette-role and pure-transpose coverage")
