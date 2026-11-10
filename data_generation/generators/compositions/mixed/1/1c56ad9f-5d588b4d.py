"""A unary run field whose generated rows follow a bottom-anchored wave.

Only standard library and NumPy are used. No parent modules or files are needed.
The finite-width wave uses background fill and clipping, never circular shifts.
"""

from functools import lru_cache
import hashlib

import numpy as np


# Input specifications, not output examples: avoid copying either B evidence set.
# A's multirow scaffold inputs cannot be confused with this family's single bar.
_PARENT_INPUTS = frozenset({
    (6, 11, 8, 0, 3), (4, 23, 7, 0, 1), (3, 5, 8, 0, 7),
    (3, 23, 7, 0, 4), (8, 16, 13, 0, 9), (7, 10, 7, 0, 6),
    (6, 21, 3, 5, 7), (6, 12, 26, 5, 9), (3, 13, 2, 5, 8),
    (3, 26, 30, 5, 6), (7, 12, 15, 5, 2),
})
_CYCLE = (0, -1, 0, 1)


def _length(n):
    return n * n + 2 * n - 2


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer rectangles of size 1..30 and colors 0..9")
    return a


def _parse(value):
    a = _grid(value)
    fg, bg = int(a[0, 0]), int(a[-1, -1])
    if fg == bg:
        raise ValueError("the upper-left bar and background must both be visible")
    w = a.shape[1]
    n = 0
    while n < w and int(a[0, n]) == fg:
        n += 1
    if (not 2 <= n <= 29 or np.any(a[0, n:] != bg)
            or np.any(a[1:] != bg) or (_length(n) + w - 1) // w > 30):
        raise ValueError("expected one upper-left horizontal bar and a legal generated height")
    return a, n, w, bg, fg


def _field(n, w):
    """Canonical unshifted boolean field, with no extra trailing blank row."""
    cells = []
    lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    for i, length in enumerate(lengths):
        if i:
            cells.append(False)
        cells.extend([True] * length)
    cells.extend([False] * (-len(cells) % w))
    return np.array(cells, dtype=bool).reshape(-1, w)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive both the unary field and its wave entirely from the visible input."""
    _, n, w, bg, fg = _parse(input_grid)
    base = _field(n, w)
    h = base.shape[0]
    result = np.full((h, w), bg, dtype=np.int8)
    for r in range(h):
        columns = np.flatnonzero(base[r]) + _CYCLE[(h - 1 - r) % 4]
        columns = columns[(columns >= 0) & (columns < w)]
        result[r, columns] = fg
    return result


def _alternative(n, w, bg, fg, kind, input_height=None):
    """Explicit competing rules, used only to audit demonstration evidence."""
    base = _field(n, w)
    h = base.shape[0]
    out = np.full((h, w), bg, dtype=np.int8)
    for r in range(h):
        d = h - 1 - r
        if kind == "no_wave":
            offset = 0
        elif kind == "top_anchor":
            offset = _CYCLE[r % 4]
        elif kind == "input_anchor":
            offset = _CYCLE[(input_height - 1 - r) % 4]
        elif kind == "reversed":
            offset = -_CYCLE[d % 4]
        else:
            offset = _CYCLE[d % 4]
        columns = np.flatnonzero(base[r]) + offset
        if kind == "circular":
            columns %= w
        else:
            columns = columns[(columns >= 0) & (columns < w)]
        out[r, columns] = fg
    return out


@lru_cache(maxsize=512)
def _profile(n, w):
    base = _field(n, w)
    h = base.shape[0]
    left = sum(bool(base[r, 0]) for r in range(h) if (h - 1 - r) % 4 == 1)
    right = sum(bool(base[r, -1]) for r in range(h) if (h - 1 - r) % 4 == 3)
    raw = base.astype(np.int8)
    wave = _alternative(n, w, 0, 1, "bottom")
    cuts = range(w, _length(n), w)
    flat = base.ravel()
    inside = any(flat[p - 1] and flat[p] for p in cuts)
    separator = any(not flat[p - 1] or not flat[p] for p in cuts)
    peak_start = n * (n - 1) // 2 + n - 1
    peak = any(peak_start < p < peak_start + n for p in cuts)
    return (h, (-_length(n)) % w, left, right,
            not np.array_equal(raw, wave),
            not np.array_equal(wave, _alternative(n, w, 0, 1, "top_anchor")),
            inside, separator, peak)


def _geometries(ns, min_height=1, max_height=30, excluded_phase=None):
    return [(n, w) for n in ns for w in range(n, 31)
            if min_height <= (_length(n) + w - 1) // w <= max_height
            and (excluded_phase is None or _profile(n, w)[0] % 4 != excluded_phase)]


def _pick(rng, choices):
    if not choices:
        raise RuntimeError("empty geometry schedule")
    return choices[int(rng.integers(len(choices)))]


def _choose_geometry(rng, candidates, goal=None):
    if goal == "exact":
        preferred = [p for p in candidates if _profile(*p)[1] == 0]
    elif goal == "one_pad":
        preferred = [p for p in candidates if _profile(*p)[1] == 1]
    elif goal == "long_pad":
        preferred = [p for p in candidates if _profile(*p)[1] >= p[1] // 2]
    elif goal == "peak_cut":
        preferred = [p for p in candidates if _profile(*p)[8]]
    elif goal == "full_bar":
        preferred = [p for p in candidates if p[0] == p[1]]
    else:
        preferred = candidates
    return _pick(rng, preferred or candidates)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Sample one fixed rule with length, wrapping, or phase generalization."""
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    small = int(rng.integers(3, 5))
    large = int(rng.integers(6, 12))
    first = _pick(rng, _geometries([small], 1, 1))
    anchors = [p for p in _geometries([large], 5, 13)
               if p[1] != first[1] and _profile(*p)[2] > 0
               and _profile(*p)[3] > 0 and _profile(*p)[5]]
    second = _pick(rng, anchors)
    held_n = int(rng.integers(small + 1, large)) if regime == 1 else None
    used_phases = {1, _profile(*second)[0] % 4}
    held_phase = _pick(rng, [p for p in range(4) if p not in used_phases])
    extra_pool = _geometries([n for n in range(2, large + 1) if n != held_n],
                             excluded_phase=held_phase if regime == 3 else None)
    plans = [first, second]
    goals = ("free", "exact", "one_pad", "long_pad", "peak_cut", "full_bar")
    while len(plans) < nt:
        plans.append(_choose_geometry(rng, extra_pool, _pick(rng, goals)))

    # Literal colors have no puzzle-wide meaning. Some puzzles use one background.
    stable_background = bool(rng.random() < 0.30)
    base_background = int(rng.integers(10))
    previous_foregrounds = []
    seen = set()

    def materialize(nw, anchor=False, same_colors=None, avoid_height=None):
        n, w = nw
        h = _profile(n, w)[0]
        if same_colors is None:
            bg = base_background if stable_background else int(rng.integers(10))
            reusable = [f for f in previous_foregrounds if f != bg]
            fg = (_pick(rng, reusable) if reusable and rng.random() < 0.40
                  else _pick(rng, [c for c in range(10) if c != bg]))
        else:
            bg, fg = same_colors
        heights = [ih for ih in range(1 if n < w else 2, 31)
                   if ih != avoid_height and (not anchor or ih % 4 != h % 4)]
        rng.shuffle(heights)
        # At most 14 inputs are retained. There are at least 20 heights for an
        # anchor and 29 otherwise, so this finite search cannot exhaust uniqueness.
        for ih in heights:
            ih = int(ih)
            spec = (n, w, ih, bg, fg)
            if spec in seen or spec in _PARENT_INPUTS:
                continue
            inp = np.full((ih, w), bg, dtype=np.int8)
            inp[0, :n] = fg
            if anchor and np.array_equal(solve(inp),
                    _alternative(n, w, bg, fg, "input_anchor", ih)):
                continue
            seen.add(spec)
            previous_foregrounds.append(fg)
            return {"input": inp, "output": solve(inp)}
        raise RuntimeError("unique input height schedule exhausted")

    train = [materialize(p, anchor=(i == 1)) for i, p in enumerate(plans)]
    # A controlled height-only contrast is optional once both core anchors exist.
    if nt >= 3 and rng.random() < 0.30:
        _, n, w, bg, fg = _parse(train[0]["input"])
        train[-1] = materialize((n, w), same_colors=(bg, fg),
                                avoid_height=train[0]["input"].shape[0])
    rng.shuffle(train)
    train_specs = [_parse(p["input"])[1:] for p in train]
    train_nws = {(s[0], s[1]) for s in train_specs}
    if regime == 0:
        # Sample length before width, so the fewer legal widths of long bars do
        # not make the upper end of the supported length range nearly invisible.
        test_n = int(rng.integers(max(s[0] for s in train_specs) + 1, 30))
        candidates = _geometries([test_n], 2)
    elif regime == 1:
        candidates = _geometries([held_n], 2)
    elif regime == 2:
        candidates = [p for p in _geometries([large], 2) if p not in train_nws]
    else:
        candidates = [p for p in _geometries(range(2, 30), 2)
                      if _profile(*p)[0] % 4 == held_phase]
    candidates = [p for p in candidates if _profile(*p)[4]]
    test_plans = [_choose_geometry(rng, candidates, _pick(rng, goals))]
    all_options = _geometries(range(2, 30))
    while len(test_plans) < ne:
        # Distinct geometry, with occasional single-row specializations.
        options = [p for p in all_options if p not in test_plans and p not in train_nws]
        test_plans.append(_choose_geometry(rng, options, _pick(rng, goals)))
    puzzle = {"train": train, "test": [materialize(p) for p in test_plans]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _audit_pair(pair):
    """Check output by run intervals and source-to-destination coordinates.

    This does not call solve or _field: it also checks the pre-wave foreground
    budget, exact losses at each edge, every row, and final padding destinations.
    """
    _assert(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    inp, n, w, bg, fg = _parse(pair["input"])
    out = _grid(pair["output"])
    h = (_length(n) + w - 1) // w
    _assert(out.shape == (h, w), "wrong minimal generated canvas")
    expected = np.full((h, w), bg, dtype=np.int8)
    cursor = foreground_count = lost = 0
    losses = [0] * h
    source_counts = [0] * h
    for k in range(2 * n - 1):
        length = n - abs(n - 1 - k)
        for position in range(cursor, cursor + length):
            row, column = divmod(position, w)
            source_counts[row] += 1
            d = h - 1 - row
            delta = 0 if d % 2 == 0 else (-1 if d % 4 == 1 else 1)
            target = column + delta
            foreground_count += 1
            if 0 <= target < w:
                expected[row, target] = fg
            else:
                lost += 1
                losses[row] += 1
        cursor += length + (k < 2 * n - 2)
    _assert(cursor == _length(n) and foreground_count == n * n, "unary accounting")
    _assert(np.array_equal(out, expected), "wrong run cells, row phase, edge fill, or padding")
    _assert(int(np.count_nonzero(out == fg)) == n * n - lost, "wrong clipped foreground budget")
    _assert(all(int(np.count_nonzero(out[r] == fg)) == source_counts[r] - losses[r]
                for r in range(h)), "row translation changed its retained foreground")
    _assert(np.all((out == bg) | (out == fg)), "color role changed")
    _assert((n, w, inp.shape[0], bg, fg) not in _PARENT_INPUTS, "parent input copied")
    return inp, n, w, bg, fg


def validate_generated_puzzle(puzzle):
    """Validate cells, both parent necessities, and training disambiguation."""
    try:
        _assert(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
        _assert(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split schema")
        nt, ne = len(puzzle["train"]), len(puzzle["test"])
        _assert(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
        seen = set()
        info = {}
        for split in ("train", "test"):
            info[split] = []
            for pair in puzzle[split]:
                spec = _audit_pair(pair)
                inp = spec[0]
                key = (inp.shape, inp.astype(np.int8).tobytes())
                _assert(key not in seen, "duplicate input")
                seen.add(key)
                info[split].append(spec)
        train = info["train"]
        _assert(any(_profile(n, w)[0] == 1 and n >= 3 for _, n, w, _, _ in train),
                "training needs a readable unshifted symmetric unary demonstration")
        _assert(any(_profile(n, w)[0] >= 5 and _profile(n, w)[2] and _profile(n, w)[3]
                    for _, n, w, _, _ in train), "training must expose all phases and both clipping edges")
        _assert(len({n for _, n, _, _, _ in train}) >= 2, "training needs different lengths")
        _assert(len({w for _, _, w, _, _ in train}) >= 2, "training needs different wraps")
        for kind in ("no_wave", "top_anchor", "input_anchor", "reversed", "circular"):
            _assert(any(not np.array_equal(pair["output"],
                        _alternative(n, w, bg, fg, kind, inp.shape[0]))
                        for pair, (inp, n, w, bg, fg) in zip(puzzle["train"], train)),
                    f"training permits alternative: {kind}")
        # Without B, A sees just the input bar at distance zero: it is unchanged.
        _assert(any(not np.array_equal(p["input"], p["output"]) for p in puzzle["train"]),
                "omitting B does not change training")
        _assert(any(not np.array_equal(p["output"], _alternative(n, w, bg, fg, "no_wave"))
                    for p, (_, n, w, bg, fg) in zip(puzzle["test"], info["test"])),
                "test split must require A")
        _assert(any(not np.array_equal(p["input"], p["output"]) for p in puzzle["test"]),
                "test split must require B")
    except (ValueError, TypeError, IndexError) as error:
        raise AssertionError(f"malformed puzzle: {error}") from error


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, in-memory checks; no logs, samples, or traces are written."""
    signatures, structures, shapes, ns, widths, heights = set(), set(), set(), set(), set(), set()
    bg_colors, fg_colors, phases, pads, cuts = set(), set(), set(), set(), set()
    losses = full_bars = 0
    shifts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        _assert(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, n, w, bg, fg = _audit_pair(pair)
                _assert(np.array_equal(solve(inp), pair["output"]), "solve mismatch")
                ns.add(n); widths.add(w); heights.add(inp.shape[0])
                bg_colors.add(bg); fg_colors.add(fg)
                h, pad, left, right, _, _, inside, separator, peak = _profile(n, w)
                phases.add(h % 4)
                pads.add("exact" if pad == 0 else "one" if pad == 1 else "long" if pad >= 5 else "other")
                if inside: cuts.add("inside")
                if separator: cuts.add("separator")
                if peak: cuts.add("peak")
                full_bars += n == w
                losses += left + right
                structures.add((pair["output"].shape, (pair["output"] == fg).tobytes()))
                shapes.add((inp.shape, pair["output"].shape))
                # Input height and literal palette must carry no hidden information.
                changed = np.full((2 if n == w else 1, w), bg, dtype=np.int8)
                changed[0, :n] = fg
                _assert(np.array_equal(solve(changed), pair["output"]), "input-height dependence")
        train = [_parse(p["input"])[1:3] for p in puzzle["train"]]
        test = [_parse(p["input"])[1:3] for p in puzzle["test"]]
        if any(n > max(x[0] for x in train) for n, _ in test): shifts.add("extrapolation")
        if any(n not in {x[0] for x in train} and min(x[0] for x in train) < n < max(x[0] for x in train)
               for n, _ in test): shifts.add("interpolation")
        if any(n in {x[0] for x in train} and (n, w) not in train for n, w in test): shifts.add("rewrap")
        if any(_profile(n, w)[0] % 4 not in {_profile(*x)[0] % 4 for x in train} for n, w in test):
            shifts.add("new_phase")
    _assert(len(signatures) == 42 and len(structures) >= 70, "too little structural variation")
    _assert(len(ns) >= 15 and max(ns) >= 24 and len(widths) >= 20, "length/width range too narrow")
    _assert(len(heights) >= 25 and len(shapes) >= 100, "shape variation too narrow")
    _assert(len(bg_colors) == len(fg_colors) == 10, "palette role range too narrow")
    _assert(phases == {0, 1, 2, 3} and {"exact", "one", "long"} <= pads, "wrap coverage missing")
    _assert(cuts == {"inside", "separator", "peak"} and full_bars > 0 and losses > 0, "boundary coverage missing")
    _assert(shifts == {"extrapolation", "interpolation", "rewrap", "new_phase"}, "shift regimes missing")
    legal_count_calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1700 + nt * 10 + ne, nt, ne))
            legal_count_calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exhaust the entire supported (n,width) domain, including one-row inputs,
    # full-width bars, single-row fields, and near-30x30 fields.
    geometries = _geometries(range(2, 30))
    for n, w in geometries:
        inp = np.full((2 if n == w else 1, w), 7, dtype=np.int8)
        inp[0, :n] = 0
        before = inp.copy()
        output = solve(inp)
        _audit_pair({"input": inp, "output": output})
        _assert(np.array_equal(inp, before), "input mutated")
        recolored = np.where(inp == 0, 9, 2)
        _assert(np.array_equal(solve(recolored), np.where(output == 0, 9, 2)), "palette dependence")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "output_structures": len(structures), "shape_combinations": len(shapes),
            "lengths": sorted(ns), "widths": sorted(widths), "input_heights": sorted(heights),
            "phases": sorted(phases), "padding_classes": sorted(pads),
            "cut_classes": sorted(cuts), "full_width_bars": full_bars,
            "clipped_cells": losses, "shift_regimes": sorted(shifts),
            "legal_count_calls": legal_count_calls, "exhaustive_geometries": len(geometries)}
