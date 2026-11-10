"""An elastic period-six ruler: equality bands supply its distance metric.

Only Python's standard library and NumPy are used. No files, generation metadata,
or demonstrations are consulted by solve. All diagnostic work is in memory.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np


HIGHLIGHT = 3
SIDES = ("top", "right", "bottom", "left")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _canonical(a, side):
    """Seed edge first, with rows increasing inward; lateral order is retained."""
    if side == "top":
        return a
    if side == "right":
        return a.T[::-1]
    if side == "bottom":
        return a[::-1]
    if side == "left":
        return a.T
    raise ValueError("unknown side")


def _orient(a, side):
    if side == "right":
        return a[::-1].T.copy()
    return _canonical(a, side).copy()


def _starts(a, axis):
    if axis == 0:
        changes = np.any(a[1:] != a[:-1], axis=1)
    else:
        changes = np.any(a[:, 1:] != a[:, :-1], axis=0)
    return np.r_[0, np.flatnonzero(changes) + 1].astype(int)


def _decode(input_grid):
    """Locate the docking edge, then compare full ORIGINAL body vectors.

    The seed's one-pixel edge line is logical distance zero. The filled mosaic
    occupies every remaining inward pixel row, between uniform side gutters.
    It is the domain for both equality comparisons; the seed is excluded.
    """
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (a[0, -1] == a[-1, 0] == a[-1, -1] == bg):
        raise ValueError("the four corners must identify one background")
    if HIGHLIGHT in a:
        raise ValueError("green (3) is reserved for output highlighting")
    candidates = []
    for side in SIDES:
        g = _canonical(a, side)
        if min(g.shape) < 3:
            continue
        edge = np.flatnonzero(g[0] != bg)
        if not len(edge) or not np.array_equal(edge, np.arange(edge[0], edge[-1] + 1)):
            continue
        start, end = int(edge[0]), int(edge[-1]) + 1
        bar = int(g[0, start])
        if not np.all(g[0, start:end] == bar) or np.any(g[1:] == bar):
            continue
        mask = g[1:] != bg
        cols = np.flatnonzero(np.any(mask, axis=0))
        if not len(cols):
            continue
        left, right = int(cols[0]), int(cols[-1]) + 1
        if not (0 < left <= start < end <= right < g.shape[1]):
            continue
        if not np.all(mask[:, left:right]) or np.any(mask[:, :left]) or np.any(mask[:, right:]):
            continue
        body = g[1:, left:right]
        if len(np.unique(body)) < 2 or (start == left and end == right):
            continue
        rows, columns = _starts(body, 0), _starts(body, 1)
        rb = np.r_[rows, body.shape[0]]
        cb = np.r_[columns, body.shape[1]]
        if start - left not in cb or end - left not in cb:
            continue
        latent = body[np.ix_(rows, columns)]
        candidates.append(dict(side=side, background=bg, bar=bar, start=start,
                               end=end, left=left, right=right, rows=rb,
                               columns=cb, latent=latent.copy()))
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous edge bar docked to a filled mosaic")
    return a, candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, info = _decode(input_grid)
    result = _canonical(a, info["side"]).copy()
    s, e = info["start"], info["end"]
    # All boundaries were frozen before painting: destructive clearing cannot
    # merge later bands, and a separated recurrence still has its own index.
    for index, (lo, hi) in enumerate(zip(info["rows"][:-1], info["rows"][1:]), 1):
        block = result[1 + lo:1 + hi]
        phase = index % 6
        if phase == 4:
            block[:, info["left"]:info["right"]] = HIGHLIGHT
        else:
            block[:, s:e] = info["background"] if phase % 2 else info["bar"]
    return _orient(result, info["side"])


def _logical_output(latent, start, end, background, bar):
    """Generation paints logical cells BEFORE expanding them to pixel bands."""
    out = latent.copy()
    for r in range(len(out)):
        d = r + 1
        if d % 6 == 4:
            out[r] = HIGHLIGHT
        elif d % 2:
            out[r, start:end] = background
        else:
            out[r, start:end] = bar
    return out


def _visible(a):
    return (np.all(np.any(a[1:] != a[:-1], axis=1))
            and np.all(np.any(a[:, 1:] != a[:, :-1], axis=0)))


def _pattern(rng, height, width, palette, style, start, end):
    """Procedural content, with witnesses rather than a bank of solved layouts."""
    outside = [c for c in range(width) if not start <= c < end]
    for _ in range(128):
        a = rng.choice(palette, size=(height, width)).astype(np.int8)
        if style in ("recurrence", "weave"):
            r = int(rng.integers(height - 2))
            rr = int(rng.integers(r + 2, height))
            c = int(rng.integers(width - 2))
            cc = int(rng.integers(c + 2, width))
            a[rr] = a[r]
            a[:, cc] = a[:, c]
        if style == "seam":
            r = int(rng.integers(height - 1))
            c = int(rng.choice(outside))
            a[r + 1] = a[r]
            a[r + 1, c] = rng.choice([v for v in palette if v != a[r, c]])
        if style == "weave":
            # Locally equal neighboring cells and recurrent full vectors coexist.
            c = int(rng.integers(width - 1))
            rr = int(rng.integers(height))
            a[:, c + 1] = a[:, c]
            a[rr, c + 1] = rng.choice([v for v in palette if v != a[rr, c]])
        if _visible(a) and len(np.unique(a)) >= 2:
            return a
    # Constructive bounded fallback. Alternation has distinct adjacent vectors
    # and separated repeats in BOTH axes. A seam has one remote witness.
    rr, cc = np.indices((height, width))
    a = np.asarray(palette, dtype=np.int8)[(rr + cc) % 2]
    if style == "seam":
        r = int(rng.integers(height - 1))
        c = int(rng.choice(outside))
        a[r + 1] = a[r]
        a[r + 1, c] = palette[1] if a[r, c] == palette[0] else palette[0]
    assert _visible(a)
    return a


def _profile(rng, length, cap, peak=False, uneven=False, maximum=3):
    """Positive expansion factors, constructed inside a fixed pixel budget."""
    if peak:
        a = np.ones(length, dtype=int)
        p = int(rng.integers(length))
        a[p] = int(rng.integers(5, min(8, cap - length + 1) + 1))
        budget = int(rng.integers(0, cap - int(a.sum()) + 1))
        for _ in range(budget):
            choices = np.flatnonzero(a < 3)
            if not len(choices):
                break
            a[int(rng.choice(choices))] += 1
        return a
    if not uneven and rng.random() < .22:
        return np.full(length, int(rng.integers(1, min(maximum, cap // length) + 1)))
    a = rng.integers(1, maximum + 1, size=length)
    while a.sum() > cap:
        a[int(rng.choice(np.flatnonzero(a > 1)))] -= 1
    if len(set(a.tolist())) == 1:
        if a[0] > 1:
            a[int(rng.integers(length))] -= 1
        else:
            a[int(rng.integers(length))] += 1
    return a


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _alternate(a, info, mode):
    """Explicit simpler hypotheses, used for teaching-evidence checks only."""
    g = _canonical(a, info["side"])
    body = g[1:, info["left"]:info["right"]]
    if mode == "pixels":
        distance = np.arange(len(g))
    else:
        if mode == "strip_only":
            starts = _starts(g[1:, info["start"]:info["end"]], 0)
            ends = np.r_[starts[1:], len(body)]
            distance = np.r_[0, np.repeat(np.arange(1, len(starts) + 1), ends - starts)]
        elif mode == "global_unique":
            labels, sequence = {}, []
            for row in info["latent"]:
                sequence.append(labels.setdefault(row.tobytes(), len(labels) + 1))
            distance = np.r_[0, np.repeat(sequence, np.diff(info["rows"]))]
        else:
            raise ValueError("unknown alternative")
    out = g.copy()
    for r, d in enumerate(distance):
        out[r, info["start"]:info["end"]] = (
            HIGHLIGHT if d % 6 == 4 else info["background"] if d % 2 else info["bar"])
        if d % 6 == 4:
            mask = (g[r] != info["background"]) & (g[r] != info["bar"])
            out[r, mask] = HIGHLIGHT
    return _orient(out, info["side"])


def _has_recurrence(a, axis):
    vectors = a if axis == 0 else a.T
    keys = [v.tobytes() for v in vectors]
    return len(set(keys)) < len(keys)


def _evidence(a, out, info):
    latent = info["latent"]
    return dict(
        pixel_phase=not np.array_equal(out, _alternate(a, info, "pixels")),
        remote_seam=not np.array_equal(out, _alternate(a, info, "strip_only")),
        separated_phase=not np.array_equal(out, _alternate(a, info, "global_unique")),
        row_recurrence=_has_recurrence(latent, 0),
        column_recurrence=_has_recurrence(latent, 1),
        uneven_rows=len(set(np.diff(info["rows"]).tolist())) > 1,
        uneven_columns=len(set(np.diff(info["columns"]).tolist())) > 1,
        cycle=len(latent) >= 12)


def _sample_pair(rng, side, style, split, first=False):
    height = int(rng.integers(12, 16) if first else rng.integers(7, 16))
    if split == "test":
        height = int(rng.integers(6, 19))
    width = int(rng.integers(4, 9))
    length = int(rng.integers(1, width))
    start = int(rng.integers(width - length + 1))
    end = start + length
    colors = rng.permutation([c for c in range(10) if c != HIGHLIGHT]).tolist()
    bg, bar = colors[:2]
    palette = colors[2:2 + int(rng.integers(2, 7))]
    latent = _pattern(rng, height, width, palette, style, start, end)
    rows = _profile(rng, height, 29, peak=split == "test", uneven=style in ("seam", "recurrence"))
    columns = _profile(rng, width, 28, uneven=style in ("seam", "recurrence"), maximum=5)
    spare = 30 - int(columns.sum())
    total = int(rng.integers(2, min(8, spare) + 1))
    left = int(rng.integers(1, total))
    right = total - left
    expanded = np.repeat(np.repeat(latent, rows, axis=0), columns, axis=1)
    shape = (1 + int(rows.sum()), left + expanded.shape[1] + right)
    a = np.full(shape, bg, dtype=np.int8)
    b = a.copy()
    a[1:, left:shape[1] - right] = expanded
    edited = _logical_output(latent, start, end, bg, bar)
    b[1:, left:shape[1] - right] = np.repeat(np.repeat(edited, rows, axis=0), columns, axis=1)
    cb = np.r_[0, np.cumsum(columns)]
    a[0, left + cb[start]:left + cb[end]] = bar
    b[0] = a[0]
    return {"input": _orient(a, side), "output": _orient(b, side)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require integer counts 2..10 / 1..4 with num_train > num_test")
    rng = np.random.default_rng(seed)
    base = int(rng.integers(4))
    turn = int(rng.choice([1, 3]))
    training_sides = [SIDES[base], SIDES[(base + turn) % 4]]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            side = training_sides[index % 2]
            if split == "test":
                side = SIDES[(SIDES.index(side) + 2) % 4]
            style = ("seam" if index == 0 else "recurrence") if split == "train" and index < 2 else str(
                rng.choice(["random", "seam", "recurrence", "weave"]))
            for _ in range(64):
                pair = _sample_pair(rng, side, style, split, first=split == "train" and index == 0)
                a, info = _decode(pair["input"])
                evidence = _evidence(a, pair["output"], info)
                if _signature(a) in seen:
                    continue
                if split == "train" and index == 0 and not (evidence["remote_seam"] and evidence["pixel_phase"]):
                    continue
                if split == "train" and index == 1 and not (evidence["separated_phase"]
                        and evidence["row_recurrence"] and evidence["column_recurrence"]):
                    continue
                seen.add(_signature(a))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded construction could not find distinct teaching evidence")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pixel_check(a, info):
    """Direct per-pixel oracle: group coordinates, with no block painting."""
    g = _canonical(a, info["side"])
    out = g.copy()
    for r in range(1, len(g)):
        d = int(np.searchsorted(info["rows"], r - 1, side="right"))
        for c in range(g.shape[1]):
            if info["start"] <= c < info["end"]:
                out[r, c] = (HIGHLIGHT if d % 6 == 4 else
                             info["background"] if d % 2 else info["bar"])
            elif d % 6 == 4 and g[r, c] != info["background"]:
                out[r, c] = HIGHLIGHT
    return _orient(out, info["side"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, evidence, train_sides = set(), [], set()
    test_infos = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, info = _decode(pair["input"])
            b = _grid(pair["output"])
            assert a.shape == b.shape and not np.array_equal(a, b)
            assert _signature(a) not in seen
            seen.add(_signature(a))
            assert np.array_equal(b, solve(a))
            assert np.array_equal(b, _pixel_check(a, info)), "incorrect band or cross-section edit"
            rows, cols = np.diff(info["rows"]), np.diff(info["columns"])
            g = _canonical(a, info["side"])
            reconstruction = np.repeat(np.repeat(info["latent"], rows, axis=0), cols, axis=1)
            assert np.array_equal(reconstruction, g[1:, info["left"]:info["right"]])
            assert _visible(info["latent"])
            if split == "train":
                evidence.append(_evidence(a, b, info))
                train_sides.add(info["side"])
                assert max(rows) <= 3
            else:
                test_infos.append(info)
                assert max(rows) >= 5
    # Requirements describe demonstrations and this sampler's shift, not extra
    # information needed by solve. Both compression axes are taught physically.
    assert len(train_sides) == 2
    side_ids = sorted(SIDES.index(s) for s in train_sides)
    assert (side_ids[1] - side_ids[0]) % 4 in (1, 3)
    assert all(i["side"] not in train_sides for i in test_infos)
    for name in ("pixel_phase", "remote_seam", "separated_phase", "row_recurrence",
                 "column_recurrence", "uneven_rows", "uneven_columns", "cycle"):
        assert any(e[name] for e in evidence), "missing training witness: " + name


def _digest_puzzle(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, streamed: retain small hash sets and aggregate metrics only."""
    signatures, inputs, shapes, logical_shapes, profiles, structures = (set() for _ in range(6))
    split_sides = {"train": set(), "test": set()}
    backgrounds, markers, palettes, lengths = (set() for _ in range(4))
    counts = {k: 0 for k in ("pairs", "row_recurrence", "column_recurrence", "remote_seam",
                             "pixel_phase", "separated_phase", "single_pixel_bar")}
    for seed in range(42):
        p = generate_puzzle(seed)
        assert _digest_puzzle(p) == _digest_puzzle(generate_puzzle(seed))
        signatures.add(_digest_puzzle(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a, info = _decode(pair["input"])
                latent = info["latent"]
                inputs.add(hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest())
                shapes.add(a.shape)
                logical_shapes.add(latent.shape)
                profiles.add((tuple(np.diff(info["rows"])), tuple(np.diff(info["columns"]))))
                split_sides[split].add(info["side"])
                backgrounds.add(info["background"])
                markers.add(info["bar"])
                palettes.add(len(np.unique(latent)))
                lengths.add(info["end"] - info["start"])
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in latent.flat)
                structures.add((latent.shape, normalized))
                e = _evidence(a, pair["output"], info)
                counts["pairs"] += 1
                for name in ("row_recurrence", "column_recurrence", "remote_seam", "pixel_phase", "separated_phase"):
                    counts[name] += int(e[name])
                counts["single_pixel_bar"] += info["end"] - info["start"] == 1
                # Compression order commutes because replication preserves every
                # original inequality witness. Compare row-first and column-first.
                body = _canonical(a, info["side"])[1:, info["left"]:info["right"]]
                row_first = body[_starts(body, 0)]
                row_first = row_first[:, _starts(row_first, 1)]
                col_first = body[:, _starts(body, 1)]
                col_first = col_first[_starts(col_first, 0)]
                assert np.array_equal(row_first, col_first) and np.array_equal(row_first, latent)
    assert len(signatures) == 42 and len(inputs) == counts["pairs"] == 210
    assert len(shapes) >= 65 and len(logical_shapes) >= 35 and len(structures) >= 150
    assert len(profiles) >= 150 and all(len(v) == 4 for v in split_sides.values())
    assert len(backgrounds) == len(markers) == 9 and len(palettes) >= 4
    assert len(lengths) >= 15 and counts["single_pixel_bar"] > 0
    assert all(counts[k] >= 42 for k in ("remote_seam", "pixel_phase", "separated_phase"))
    # All legal explicit count combinations; no puzzles are retained between calls.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    # Hand-worked regression, independent of the sampler. Logical distances
    # 1/2 agree inside the strip but differ remotely; distance 4 repeats row 1.
    # Unequal replication must survive in the output rather than being cropped.
    query = "00010000 02224420 02224420 02224450 04452240 04452240 02224420 05545520 04452240"
    answer = "00010000 02204420 02204420 02214450 04402240 04402240 03333330 05505520 04412240"
    query = np.array([[int(c) for c in row] for row in query.split()], dtype=np.int8)
    answer = np.array([[int(c) for c in row] for row in answer.split()], dtype=np.int8)
    assert np.array_equal(solve(query), answer)
    # Solver covariance and validator sensitivity, including the far background.
    p = generate_puzzle(314159, 2, 1)
    pair = p["train"][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
    assert np.array_equal(solve(pair["input"][:, ::-1]), pair["output"][:, ::-1])
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), pairs=counts["pairs"],
                input_shapes=len(shapes), logical_shapes=len(logical_shapes),
                color_normalized_mosaics=len(structures), profiles=len(profiles),
                evidence=counts, independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
