"""Reflection-supported line signatures followed by adjacent-vector collapse.

Only standard Python and NumPy are used. No files, seed, or demonstrations are
consulted by solve(). Vertical reflection is a fixed family convention.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= a.size:
        raise ValueError("the background must occupy a strict majority")
    return bg


def _scores(mask):
    # Convolution counts occupied cells supported about each doubled axis.
    # An off-axis pair contributes two; an on-axis cell contributes one.
    result = np.zeros(2 * mask.shape[1] - 1, dtype=np.int32)
    for row in mask:
        result += np.convolve(row.astype(np.int32), row.astype(np.int32))
    return result


def _clean(a):
    bg = _background(a)
    clean = np.full_like(a, bg)
    axes = {}
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        mask = a == color
        scores = _scores(mask)
        winners = np.flatnonzero(scores == scores.max())
        if len(winners) != 1:
            raise ValueError(f"color {color} has tied maximum-support vertical axes")
        axis = int(winners[0])
        reflected = axis - np.arange(a.shape[1])
        valid = (reflected >= 0) & (reflected < a.shape[1])
        keep = np.zeros_like(mask)
        keep[:, valid] = mask[:, valid] & mask[:, reflected[valid]]
        clean[keep] = color
        axes[color] = axis
    return clean, axes


def _runs(a, axis):
    vectors = a if axis == 0 else a.T
    starts = np.r_[0, 1 + np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1))]
    lengths = np.diff(np.r_[starts, len(vectors)])
    return starts, lengths


def _collapse(a):
    rows, _ = _runs(a, 0)
    cols, _ = _runs(a, 1)
    return a[np.ix_(rows, cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase unmatched cells at each color's unique best axis, then contract runs."""
    return _collapse(_clean(_grid(input_grid))[0])


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _positions(rng, width, count, parity):
    for _ in range(64):
        cols = np.sort(rng.choice(np.arange(2, width - 2), count, replace=False))
        if (np.all(np.diff(cols) >= 3)
                and (parity is None or (cols[0] + cols[1]) % 2 == parity)):
            return list(map(int, cols))
    # Guaranteed legal for width >= 3*count+7, including both requested parities.
    second = 6 if parity == 0 else 5
    return [2] + [second + 3 * i for i in range(count - 1)]


def _row_lengths(rng, h, teach, shift, testing):
    lengths = np.ones(2 * h + 1, dtype=int)
    lengths[::2] = rng.integers(1, 4, h + 1)
    if teach is not None:
        lengths[0] = 2 + int(rng.integers(2))
        # A colored band in the second demonstration rules out gap-only thinning.
        if teach == 1:
            lengths[3] = 3
    else:
        lengths[1::2] += rng.choice(3, h, p=(.70, .18, .12))
    protected = None
    if shift == "spacing" and testing:
        protected = 2 * int(rng.integers(h + 1))
        lengths[protected] = int(rng.integers(5, 9))
    while int(lengths.sum()) > 30:
        candidates = [i for i, n in enumerate(lengths) if n > 1 and i != protected]
        lengths[candidates[int(rng.integers(len(candidates)))]] -= 1
    return lengths


def _base(rng, h, v, bg, teach, parity, shift, testing, fallback_width=None):
    width = (int(rng.integers(3 * v + 7, 31))
             if fallback_width is None else fallback_width)
    cols = _positions(rng, width, v, parity if teach is not None else None)
    palette = list(map(int, rng.choice([c for c in range(10) if c != bg], h + v, replace=False)))
    hc, vc = palette[:h], palette[h:]
    small = np.full((2 * h + 1, width), bg, dtype=np.int8)
    for j, (col, color) in enumerate(zip(cols, vc)):
        if (teach is not None and j < 2) or rng.random() < .65 or h == 1:
            start, end = 0, 2 * h
        else:
            start = int(rng.integers(0, max(1, 2 * h - 2)))
            end = int(rng.integers(start + 3, 2 * h + 1))
        small[start:end + 1, col] = color
    patterns = []
    for i in range(h):
        row = 2 * i + 1
        if teach is not None and i == 0:
            pad = int(rng.integers(1, min(cols[0], width - 1 - cols[1]) + 1))
            left, right = cols[0] - pad, cols[1] + pad
            holes = {cols[0], cols[1]}
        elif teach is not None and i == 1:
            left, right = 0, width - 1
            if left + right == patterns[0][2]:
                left = 1
            holes = set()
        else:
            left = int(rng.integers(0, width - 5))
            right = int(rng.integers(left + 5, width))
            if v >= 2 and rng.random() < .5:
                j, k = sorted(map(int, rng.choice(v, 2, replace=False)))
                pad = int(rng.integers(1, min(cols[j], width - 1 - cols[k]) + 1))
                left, right = cols[j] - pad, cols[k] + pad
            axis = left + right
            active = {c for c in cols if left <= c <= right and small[row, c] != bg}
            holes = set()
            for col in sorted(active):
                mate = axis - col
                if mate in active and col <= mate and rng.random() < .55:
                    holes.update((col, mate))
        axis = left + right
        mask = np.zeros(width, dtype=bool)
        mask[left:right + 1] = True
        mask[list(holes)] = False
        if int(mask.sum()) < 4:
            # Retaining the center crossing is a symmetric ownership choice.
            holes = {c for c in holes if 2 * c != axis}
            mask[left:right + 1] = True
            mask[list(holes)] = False
        if i >= 2 and rng.random() < .24:
            # Reuse a base horizontal mask and its axis. Later perforations can
            # vary supported rows. An absent rail leaves a matched background hole.
            earlier = int(rng.integers(len(patterns)))
            old_mask, old_color, axis = patterns[earlier]
            mask, hc[i] = old_mask.copy(), old_color
        small[row, mask] = hc[i]
        patterns.append((mask.copy(), hc[i], axis))
    lengths = _row_lengths(rng, h, teach, shift, testing)
    clean = np.repeat(small, lengths, axis=0)
    eligible = []
    for i, (mask, color, axis) in enumerate(patterns):
        if lengths[2 * i + 1] < 3:
            continue
        orbits = [(c, axis - c) for c in range(width)
                  if c < axis - c < width and mask[c] and mask[axis - c]
                  and c not in cols and axis - c not in cols]
        if len(orbits) >= 2:
            eligible.append((i, orbits))
    if teach == 1 or eligible:
        # Two unequal symmetric signatures with equal ink counts. The latter
        # signature is repeated: A,B,B must collapse to A,B, never to one row.
        # Holes avoid every rail, so displayed crossings are unchanged.
        if teach == 1:
            index, orbits = next(item for item in eligible if item[0] == 1)
        else:
            index, orbits = eligible[int(rng.integers(len(eligible)))]
        chosen = rng.choice(len(orbits), 2, replace=False)
        start = int(lengths[:2 * index + 1].sum())
        clean[start, list(orbits[int(chosen[0])])] = bg
        clean[start + 1:start + 3, list(orbits[int(chosen[1])])] = bg
    # This is a construction invariant, not a call to solve() for ground truth.
    axes = {color: 2 * col for color, col in zip(vc, cols)}
    axes.update({color: axis for _, color, axis in patterns})
    assert all(np.any(clean == color) for color in axes)
    return clean, axes, set(hc), set(vc)


def _noise(rng, clean, axes, horizontal, vertical, amount):
    """Add bounded unmatched noise; certify every addition with support scores."""
    a = clean.copy()
    width = a.shape[1]
    masks = {color: a == color for color in axes}
    scores = {color: _scores(mask) for color, mask in masks.items()}
    totals = {color: int(mask.sum()) for color, mask in masks.items()}
    bg = _background(clean)
    if int(np.count_nonzero(clean != bg)) + amount >= clean.size / 2:
        return None

    def add(color, candidates):
        mask, current = masks[color], scores[color]
        for index in rng.permutation(len(candidates)):
            row, col = map(int, candidates[int(index)])
            if a[row, col] != bg or 2 * col == axes[color]:
                continue
            mate = axes[color] - col
            if 0 <= mate < width and mask[row, mate]:
                continue
            delta = 2 * np.bincount(col + np.flatnonzero(mask[row]), minlength=2 * width - 1)
            delta[2 * col] += 1
            updated = current + delta
            if (updated[axes[color]] != totals[color]
                    or np.count_nonzero(updated >= totals[color]) != 1):
                continue
            a[row, col] = color
            mask[row, col] = True
            scores[color] = updated
            return True
        return False

    # A single noise cell breaks a previously identical neighbor on BOTH axes.
    row_equal = np.zeros(a.shape[0], dtype=bool)
    col_equal = np.zeros(width, dtype=bool)
    re = np.all(clean[1:] == clean[:-1], axis=1)
    ce = np.all(clean[:, 1:] == clean[:, :-1], axis=0)
    row_equal[:-1] |= re
    row_equal[1:] |= re
    col_equal[:-1] |= ce
    col_equal[1:] |= ce
    candidates = np.argwhere((a == bg) & row_equal[:, None] & col_equal[None, :])
    if not any(add(int(c), candidates) for c in rng.permutation(sorted(horizontal))):
        return None
    # An attached protrusion on a rail rules out deleting isolated components.
    attached = False
    for color in rng.permutation(sorted(vertical)):
        color = int(color)
        near = np.zeros_like(a, dtype=bool)
        mask = masks[color]
        near[:, 1:] |= mask[:, :-1]
        near[:, :-1] |= mask[:, 1:]
        near[1:] |= mask[:-1]
        near[:-1] |= mask[1:]
        if add(color, np.argwhere(near & (a == bg))):
            attached = True
            break
    if not attached:
        return None
    for _ in range(amount - 2):
        candidates = np.argwhere(a == bg)
        if not any(add(int(c), candidates) for c in rng.permutation(sorted(axes))):
            return None
    return a


def _pair(rng, h, v, bg, teach, parity, shift, testing, amount, fallback_width=None):
    clean, axes, horizontal, vertical = _base(
        rng, h, v, bg, teach, parity, shift, testing, fallback_width)
    try:
        a = _noise(rng, clean, axes, horizontal, vertical, amount)
    except ValueError:
        return None
    if a is None:
        return None
    # Check the causal dependency, including dimension witnesses for wrong order.
    out = _collapse(clean)
    raw = _collapse(a)
    if not (raw.shape[0] > out.shape[0] and raw.shape[1] > out.shape[1]):
        return None
    return {"input": a, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("spacing", "inventory", "corruption")))
    parities = list(map(int, rng.permutation(2)))
    base_bg = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        testing = split == "test"
        for i in range(count):
            teach = i if not testing and i < 2 else None
            bg = (base_bg if teach == 0 else
                  (base_bg + int(rng.integers(1, 10))) % 10 if teach == 1 else int(rng.integers(10)))
            if shift == "inventory":
                h = int(rng.integers(4, 6) if testing else rng.integers(2, 4))
            else:
                h = int(rng.integers(1 if testing else 2, 6 if testing else 5))
            v = int(rng.integers(2 if teach is not None else 1, min(4, 9 - h) + 1))
            amount = int(rng.integers(6, 10) if testing and shift == "corruption" else rng.integers(2, 5))
            parity = parities[i] if teach is not None else None
            for attempt in range(96):
                pair = _pair(rng, h, v, bg, teach, parity, shift, testing, amount)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                # Bounded rescue: at most 15 widths, no saved runs.
                # Two rails leave ample homogeneous columns for the witnesses.
                for width in range(16, 31):
                    pair = _pair(rng, h, 2, bg, teach, parity, shift, testing, amount, width)
                    if pair is not None and _key(pair["input"]) not in seen:
                        break
                else:
                    raise RuntimeError("bounded construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _coordinate_clean(a):
    """Validator reconstruction with coordinate sets, independent of convolution."""
    bg = _background(a)
    out = np.full_like(a, bg)
    axes = {}
    for color in set(map(int, np.unique(a))) - {bg}:
        cells = set(map(tuple, np.argwhere(a == color).tolist()))
        candidates = []
        for axis in range(2 * a.shape[1] - 1):
            supported = {p for p in cells if (p[0], axis - p[1]) in cells}
            candidates.append((len(supported), axis, supported))
        best = max(item[0] for item in candidates)
        winners = [item for item in candidates if item[0] == best]
        assert len(winners) == 1, "ambiguous axis"
        _, axis, supported = winners[0]
        for p in supported:
            out[p] = color
        axes[color] = axis
    return out, axes


def _global_dedup(a):
    rows, cols = [], []
    seen = set()
    for i, row in enumerate(a):
        key = tuple(map(int, row))
        if key not in seen:
            rows.append(i)
            seen.add(key)
    seen = set()
    for i, col in enumerate(a.T):
        key = tuple(map(int, col))
        if key not in seen:
            cols.append(i)
            seen.add(key)
    return a[np.ix_(rows, cols)]


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    clean, axes = _coordinate_clean(a)
    bg = _background(a)
    assert np.all((a == clean) | (clean == bg)), "cleanup must only erase"
    assert set(np.unique(clean)) == set(np.unique(a)), "a line color vanished"
    rs, rl = _runs(clean, 0)
    cs, cl = _runs(clean, 1)
    assert np.array_equal(out, clean[np.ix_(rs, cs)]), "incorrect maximal-run collapse"
    assert np.array_equal(np.repeat(np.repeat(out, rl, axis=0), cl, axis=1), clean)
    assert np.array_equal(_collapse(out), out), "output still has equal adjacent vectors"
    assert np.array_equal(solve(a), out), "input-only solver disagrees"
    raw = _collapse(a)
    assert raw.shape[0] > out.shape[0] and raw.shape[1] > out.shape[1], "noise must obstruct both axes"
    assert clean.shape[0] > out.shape[0] and clean.shape[1] > out.shape[1]
    horizontal, vertical = {}, {}
    for color, axis in axes.items():
        cells = np.argwhere(clean == color)
        columns = set(map(int, cells[:, 1]))
        if len(columns) == 1:
            col = next(iter(columns))
            assert axis == 2 * col, "on-axis rail cells must support themselves"
            vertical[color] = col
        else:
            rows = sorted(set(map(int, cells[:, 0])))
            for row in rows:
                columns = set(map(int, np.flatnonzero(clean[row] == color)))
                assert columns == {axis - c for c in columns}
            horizontal[color] = rows
    assert horizontal and vertical
    owners = set()
    line_rows = []
    colored_repeat = False
    for r in range(clean.shape[0]):
        colors = [c for c in horizontal if np.any(clean[r] == c)]
        assert len(colors) <= 1
        if not colors:
            continue
        color = colors[0]
        line_rows.append(r)
        occupied = np.flatnonzero(clean[r] == color)
        for vc, col in vertical.items():
            if occupied.min() <= col <= occupied.max():
                value = int(clean[r, col])
                assert value in (bg, color, vc), "invalid crossing color"
                if value != bg:
                    owners.add("H" if value == color else "V")
                    # Surviving crossings retain exactly the displayed input value.
                    assert a[r, col] == value
        if r and np.array_equal(clean[r], clean[r - 1]):
            colored_repeat = True
    def hcolor(row):
        return next(c for c in horizontal if np.any(clean[row] == c))

    row_bands = 1 + sum(b != a + 1 or hcolor(a) != hcolor(b)
                        for a, b in zip(line_rows, line_rows[1:]))
    position_sensitive = any(not np.array_equal(a, b)
                             and tuple(a[a != bg]) == tuple(b[b != bg])
                             for a, b in zip(clean, clean[1:]))
    spacing_runs = [int(n) for r, n in zip(rs, rl)
                    if not any(np.any(clean[r] == c) for c in horizontal)]
    deleted = a != clean
    attached = False
    for r, c in np.argwhere(deleted):
        color = a[r, c]
        attached |= any(0 <= rr < a.shape[0] and 0 <= cc < a.shape[1] and clean[rr, cc] == color
                        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))
    return {"background": bg, "axes": axes, "horizontal": horizontal,
            "vertical": vertical, "owners": owners, "noise": int(deleted.sum()),
            "h_bands": row_bands, "max_spacing": max(spacing_runs),
            "colored_repeat": colored_repeat, "attached": attached,
            "position_sensitive": position_sensitive,
            "global_dedup_fails": not np.array_equal(_global_dedup(clean), out),
            "shape": a.shape, "output_shape": out.shape,
            "row_runs": tuple(map(int, rl)), "col_runs": tuple(map(int, cl))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    analyses, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(_analyze(pair))
    first, second = analyses["train"][:2]
    for info in (first, second):
        assert info["owners"] == {"H", "V"}, "training must show both crossing owners"
        assert len({info["axes"][c] for c in info["horizontal"]}) >= 2
        assert info["attached"] and info["global_dedup_fails"]
    demonstrated = {a["axes"][c] % 2 for a in analyses["train"] for c in a["horizontal"]}
    assert demonstrated == {0, 1}, "training must demonstrate both axis parities"
    assert second["colored_repeat"], "training must contract a colored band too"
    assert second["position_sensitive"], "training must distinguish whole-vector comparison"
    assert first["background"] != second["background"]
    # At least one input-observable shift holds for ALL tests, without metadata.
    train, test = analyses["train"], analyses["test"]
    assert any(all(t[field] > max(a[field] for a in train) for t in test)
               for field in ("max_spacing", "h_bands", "noise")), "no demonstrated train/test shift"


def _puzzle_fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, all explicit counts, counterexamples and mutation checks; no I/O."""
    shapes, output_shapes, layouts, palettes, parities, counts = set(), set(), set(), set(), set(), set()
    fingerprints = set()
    no_reuse = reuse = partial_rails = mixed_axes = colored_bands = position_sensitive = 0
    shift_hits = {field: 0 for field in ("max_spacing", "h_bands", "noise")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_puzzle_fingerprint(puzzle))
        if seed < 3:
            assert _puzzle_fingerprint(puzzle) == _puzzle_fingerprint(generate_puzzle(seed))
        analyzed = {s: [_analyze(p) for p in puzzle[s]] for s in ("train", "test")}
        for field in shift_hits:
            shift_hits[field] += all(t[field] > max(a[field] for a in analyzed["train"])
                                     for t in analyzed["test"])
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], analyzed[split]):
                a = np.asarray(pair["input"])
                shapes.add(a.shape)
                output_shapes.add(np.asarray(pair["output"]).shape)
                layouts.add((info["h_bands"], len(info["vertical"]), info["row_runs"], info["col_runs"]))
                palettes.add(info["background"])
                parities.update(axis % 2 for axis in info["axes"].values())
                counts.add((info["h_bands"], len(info["vertical"])))
                reuse += info["h_bands"] > len(info["horizontal"])
                no_reuse += info["h_bands"] == len(info["horizontal"])
                mixed_axes += len(set(info["axes"].values())) > 1
                colored_bands += info["colored_repeat"]
                position_sensitive += info["position_sensitive"]
                clean = _clean(a)[0]
                partial_rails += any(np.count_nonzero(clean[:, col] == color) < a.shape[0] - len(np.flatnonzero(
                    np.any(np.isin(clean, list(info["horizontal"])), axis=1)))
                    for color, col in info["vertical"].items())
    assert len(fingerprints) == 42
    assert len(shapes) >= 50 and len(output_shapes) >= 15 and len(layouts) >= 150
    assert palettes == set(range(10)) and parities == {0, 1} and len(counts) >= 12
    assert reuse >= 5 and no_reuse >= 50 and partial_rails >= 5
    assert mixed_axes >= 150 and colored_bands >= 42
    assert position_sensitive >= 42
    assert all(n >= 3 for n in shift_hits.values())
    # Every legal explicit count combination, including the maximum 14 pairs.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, 1.5, True):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # Explicit ambiguity and on-axis support checks, including an out-of-bounds mate.
    tied = np.zeros((5, 8), dtype=int)
    tied[1, 1] = tied[3, 6] = 1
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("tied axes were accepted")
    edge = np.zeros((6, 8), dtype=int)
    edge[1:5, 0] = 2
    edge[0, 1] = 2
    clean, axes = _clean(edge)
    assert axes[2] == 0 and np.all(clean[1:5, 0] == 2) and clean[0, 1] == 0
    # Mutating a retained crossing must fail validation.
    bad = generate_puzzle(1024, 2, 1)
    out = bad["test"][0]["output"]
    bg = _background(bad["test"][0]["input"])
    r, c = np.argwhere(out != bg)[0]
    out[r, c] = bg
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("output mutation accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "layouts": len(layouts), "inventories": len(counts),
            "horizontal_reuse_pairs": reuse, "partial_rail_pairs": partial_rails,
            "colored_band_pairs": colored_bands, "position_sensitive_pairs": position_sensitive,
            "shift_puzzles": shift_hits}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
