"""Stencil looms: counted longitudinal dilation beneath a native-size stencil.

Only standard library and NumPy are used.  The parser and solve() read one input;
the constructor paints its answer from sampled objects, not by calling solve().
"""
from __future__ import annotations

import itertools

import numpy as np


# Rotations of a rightward loom, counterclockwise in image coordinates.
_DIRS = ((0, 1), (-1, 0), (0, -1), (1, 0))
_REQUIRED = frozenset((
    "no_count", "ignore_stencil", "last_slice", "no_cycle", "reverse_word", "whole_word",
    "stretch_stencil", "fuse_before_stretch", "opaque_holes", "hub_phase",
    "count_colors", "count_largest_ink", "count_bbox", "plain_extension",
))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normal(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, group = [first], {first}
        while todo:
            r, c = todo.pop()
            for dr, dc in _DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1


def _ring(h, w):
    a = np.ones((h, w), dtype=bool)
    a[1:-1, 1:-1] = False
    return a


def _coords(box, direction, distance):
    """Cross-section in the transverse order obtained by rotating a right ray."""
    r, c, h, w = box
    if direction == 0:
        return [(r + 1 + t, c + w - 1 + distance) for t in range(h - 2)]
    if direction == 1:
        return [(r - distance, c + 1 + t) for t in range(w - 2)]
    if direction == 2:
        return [(r + h - 2 - t, c - distance) for t in range(h - 2)]
    return [(r + h - 1 + distance, c + w - 2 - t) for t in range(w - 2)]


def _maximum(shape, box, direction):
    r, c, h, w = box
    return (shape[1] - c - w, r, c, shape[0] - r - h)[direction]


def _body_cells(box):
    r, c, h, w = box
    return {(i, j) for i in range(r, r + h) for j in range(c, c + w)}


def _footprint(shape, box, direction):
    return _body_cells(box) | {
        q for d in range(1, _maximum(shape, box, direction) + 1)
        for q in _coords(box, direction, d)
    }


def _collar(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _parse_color(g, bg, frame, groups):
    heads, claimed, footprints = [], set(), []
    assert 1 <= len(groups) <= 3
    for cells in groups:
        box = _bbox(cells)
        r, c, h, w = box
        assert 4 <= h <= 7 and 4 <= w <= 7
        assert np.array_equal(g[r:r+h, c:c+w] == frame, _ring(h, w))
        assert r >= 1 and c >= 1 and r + h < g.shape[0] and c + w < g.shape[1]
        choices = []
        for direction in range(4):
            if _maximum(g.shape, box, direction) < 2:
                continue
            if all(int(g[q]) not in (bg, frame) for q in _coords(box, direction, 1)):
                choices.append(direction)
        assert len(choices) == 1, "exactly one full-width carrier attaches to a frame"
        direction = choices[0]
        patch = np.rot90(g[r:r+h, c:c+w], -direction)
        stencil = patch[1:-1, 1:-1].copy()
        n, m = stencil.shape
        assert 2 <= n <= 4 and 2 <= m <= 5
        k = int(np.count_nonzero(stencil != bg))
        assert 1 <= k <= 4 and np.any(stencil == bg)
        word, seed_cells = [], set()
        maximum = _maximum(g.shape, box, direction)
        ended = False
        for d in range(1, maximum + 1):
            points = _coords(box, direction, d)
            vals = [int(g[q]) for q in points]
            if not ended and all(v not in (bg, frame) for v in vals):
                word.append(vals)
                seed_cells.update(points)
            else:
                ended = True
                assert all(v == bg for v in vals), "ray must be empty after the seed"
        assert 1 <= len(word) <= 4 and len(word) < maximum
        footprint = _footprint(g.shape, box, direction)
        assert not any(footprint & _collar(old) for old in footprints), "loom rays need empty collars"
        footprints.append(footprint)
        claimed.update(_body_cells(box) | seed_cells)
        heads.append(dict(box=box, direction=direction, stencil=stencil,
                          word=np.array(word, dtype=np.int8), k=k, maximum=maximum))
    assert all((int(r), int(c)) in claimed for r, c in zip(*np.nonzero(g != bg)))
    return heads


def _analyze_input(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert all(int(g[q]) == bg for q in ((0, -1), (-1, 0), (-1, -1)))
    interpretations = []
    for frame in (int(c) for c in np.unique(g) if c != bg):
        groups = _components(g == frame)
        if not 1 <= len(groups) <= 3:
            continue
        # Reject most payload colors before the complete geometric parse.
        good = True
        for cells in groups:
            r, c, h, w = _bbox(cells)
            if not (4 <= h <= 7 and 4 <= w <= 7 and len(cells) == 2*h + 2*w - 4):
                good = False
                break
        if good:
            try:
                interpretations.append((frame, _parse_color(g, bg, frame, groups)))
            except AssertionError:
                pass
    assert len(interpretations) == 1, "input must uniquely identify its stencil frames"
    frame, heads = interpretations[0]
    return g, bg, frame, heads


def _prefix_period(word):
    # Deliberately permissive alternative: ABA is incorrectly treated as AB.
    return next(p for p in range(1, len(word) + 1)
                if all(np.array_equal(row, word[i % p]) for i, row in enumerate(word)))


def _render(g, bg, heads, alternative=None):
    out = g.copy()
    for s in heads:
        word, stencil, k = s["word"], s["stencil"], s["k"]
        p, n = word.shape
        m = stencil.shape[1]
        if alternative in ("no_count", "plain_extension"):
            k = 1
        elif alternative == "count_colors":
            k = len(np.unique(stencil[stencil != bg]))
        elif alternative == "count_largest_ink":
            k = int(np.unique(stencil[stencil != bg], return_counts=True)[1].max())
        elif alternative == "count_bbox":
            rr, cc = np.nonzero(stencil != bg)
            k = int((rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1))
        carrier = word[::-1] if alternative == "reverse_word" else word
        if alternative == "whole_word":
            carrier = carrier[:_prefix_period(carrier)]
        j = np.arange(s["maximum"] - p)
        source = (j // k) % len(carrier)
        if alternative == "last_slice":
            source[:] = len(carrier) - 1
        elif alternative == "no_cycle":
            source = np.minimum(j // k, len(carrier) - 1)
        base = carrier[source]
        phase = j
        if alternative == "stretch_stencil":
            phase = j // k
        elif alternative == "hub_phase":
            phase = j + p
        elif alternative == "fuse_before_stretch":
            phase = (j // k) % p
        tiled = stencil[np.arange(n)[None, :], (phase % m)[:, None]]
        values = base if alternative in ("ignore_stencil", "plain_extension") else (
            tiled if alternative == "opaque_holes" else np.where(tiled == bg, base, tiled))
        for i, row in enumerate(values, p + 1):
            for q, value in zip(_coords(s["box"], s["direction"], i), row):
                out[q] = value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode one input, preserving all original material and unrelated cells."""
    g, bg, _, heads = _analyze_input(input_grid)
    return _render(g, bg, heads)


def _evidence(pair):
    g, bg, _, heads = _analyze_input(pair["input"])
    return {name for name in _REQUIRED
            if not np.array_equal(_render(g, bg, heads, name), pair["output"])}


def _sample_word(rng, p, n, palette, whole_word=False):
    inks = rng.choice(palette, size=min(len(palette), int(rng.integers(2, 5))), replace=False)
    mode = int(rng.integers(4))
    word = rng.choice(inks, size=(p, n)).astype(np.int8)
    if mode == 0:
        word[:] = rng.choice(inks, size=(p, 1))
    elif mode == 1:
        word[:] = rng.choice(inks, size=(1, n))
    elif mode == 2:
        word[:] = inks[(np.arange(p)[:, None] + np.arange(n)[None, :]) % len(inks)]
    if p > 1 and np.all(word == word[0]):
        t = int(rng.integers(n))
        word[1, t] = next(c for c in inks if c != word[1, t])
    if whole_word and p == 3:
        word[2] = word[0]
        if np.array_equal(word[0], word[1]):
            word[1, 0] = next(c for c in inks if c != word[0, 0])
    return word


def _sample_stencil(rng, n, m, k, palette, bg):
    mask = np.zeros((n, m), dtype=bool)
    style = int(rng.integers(3))
    if style == 0:
        locations = rng.choice(n*m, size=k, replace=False)
    elif style == 1:
        # Connected growth, without forcing a special corner or symmetry.
        cells = {int(rng.integers(n*m))}
        while len(cells) < k:
            frontier = sorted({(r+dr)*m + c+dc for x in cells for r, c in [divmod(x, m)]
                               for dr, dc in _DIRS
                               if 0 <= r+dr < n and 0 <= c+dc < m} - cells)
            cells.add(int(rng.choice(frontier)))
        locations = np.array(sorted(cells))
    else:
        # A compact or broken line, on independently shuffled row/column axes.
        rows, cols = rng.permutation(n), rng.permutation(m)
        locations = np.array([rows[i//m]*m + cols[i % m] for i in range(k)])
    mask.flat[locations] = True
    inks_count = int(rng.integers(1, min(k, 3, len(palette)) + 1))
    inks = rng.choice(palette, size=inks_count, replace=False)
    values = np.r_[inks, rng.choice(inks, size=k-inks_count)]
    rng.shuffle(values)
    stencil = np.full((n, m), bg, dtype=np.int8)
    stencil[mask] = values
    return stencil


def _new_head(rng, k, p, palette, bg, whole_word=False):
    n, m = int(rng.integers(2, 5)), int(rng.integers(2, 6))
    if n*m == k:
        m += 1  # A visible transparent cell is part of the family grammar.
    return dict(k=k, word=_sample_word(rng, p, n, palette, whole_word),
                stencil=_sample_stencil(rng, n, m, k, palette, bg))


def _paint(shape, heads, bg, frame):
    g = np.full(shape, bg, dtype=np.int8)
    for s in heads:
        stencil, word = s["stencil"], s["word"]
        n, m = stencil.shape
        patch = np.full((n+2, m+2), frame, dtype=np.int8)
        patch[1:-1, 1:-1] = stencil
        r, c, h, w = s["box"]
        g[r:r+h, c:c+w] = np.rot90(patch, s["direction"])
        for d, row in enumerate(word, 1):
            for q, v in zip(_coords(s["box"], s["direction"], d), row):
                g[q] = v
    out = g.copy()
    # Construction works from sampled word/stencil objects, in scalar loops.
    for s in heads:
        word, stencil, k = s["word"], s["stencil"], s["k"]
        n, m = stencil.shape
        p = len(word)
        for d in range(p+1, _maximum(shape, s["box"], s["direction"])+1):
            j = d - p - 1
            for t, q in enumerate(_coords(s["box"], s["direction"], d)):
                ink = int(stencil[t, j % m])
                out[q] = word[(j // k) % p, t] if ink == bg else ink
    return {"input": g, "output": out}


def _make_pair(rng, k, p, forbidden, direction=None, count=None, whole_word=False):
    colors = [int(c) for c in rng.permutation(10)]
    bg, frame = colors[:2]
    palette = colors[2:2+int(rng.integers(2, 9))]
    count = int(count if count is not None else rng.choice([1, 1, 2, 2, 3]))
    direction = int(rng.integers(4)) if direction is None else direction
    specs = [_new_head(rng, k, p, palette, bg, whole_word)]
    for _ in range(count - 1):
        options = [(a, b) for a in range(1, 5) for b in range(1, 5) if (a, b) != forbidden]
        a, b = options[int(rng.integers(len(options)))]
        specs.append(_new_head(rng, a, b, palette, bg))
    n, m = specs[0]["stencil"].shape
    long_min = m + 2 + p + k*p + 3
    assert long_min <= 30
    # Try free placement, including mixed axes, before a parallel packing fallback.
    for _ in range(6):
        transverse_min = max(n+4, 10 if count == 1 else 20 if count == 2 else 25)
        ch = int(rng.integers(transverse_min, 31))
        cw = int(rng.integers(long_min, 31))
        shape = (ch, cw) if direction % 2 == 0 else (cw, ch)
        placed, footprints = [], []
        for index, s in enumerate(specs):
            n, m = s["stencil"].shape
            for _attempt in range(120):
                di = direction if index == 0 else int(rng.integers(4))
                h, w = (n+2, m+2) if di % 2 == 0 else (m+2, n+2)
                r = int(rng.integers(1, shape[0]-h))
                c = int(rng.integers(1, shape[1]-w))
                box = (r, c, h, w)
                minimum = len(s["word"]) + (s["k"]*len(s["word"])+1 if index == 0 else 1)
                if _maximum(shape, box, di) < minimum:
                    continue
                footprint = _footprint(shape, box, di)
                if any(footprint & _collar(old) for old in footprints):
                    continue
                placed.append(dict(s, box=box, direction=di))
                footprints.append(footprint)
                break
            else:
                break
        if len(placed) == count:
            pair = _paint(shape, placed, bg, frame)
            _analyze_input(pair["input"])
            return pair
    # All requested heads fit in horizontal bands with one-cell empty collars.
    height = 1 + sum(s["stencil"].shape[0]+3 for s in specs)
    width = max(12, long_min, *(s["stencil"].shape[1] + 2 + len(s["word"]) + 2 for s in specs))
    width = int(rng.integers(width, 31))
    placed, row = [], 1
    for index, s in enumerate(specs):
        n, m = s["stencil"].shape
        pp = len(s["word"])
        ext = s["k"]*pp+1 if index == 0 else 1
        left_cap = width - (m+2) - pp - ext
        # Auxiliary long previews need only one new slice, always <= 14 total.
        assert left_cap >= 1
        left = int(rng.integers(1, left_cap+1))
        placed.append(dict(s, box=(row, left, n+2, m+2), direction=0))
        row += n+3
    pair = _paint((height, width), placed, bg, frame)
    return {role: np.ascontiguousarray(np.rot90(a, direction)) for role, a in pair.items()}


def _check_counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("example counts must be integers")
        if not lo <= int(value) <= hi:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _unique_pair(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    colors = np.unique(pair["input"])
    # At most 13 earlier inputs can exclude these 15 distinct recolorings.
    for values in itertools.islice(itertools.permutations(range(10), len(colors)), 15):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = values
        trial = {role: mapping[g] for role, g in pair.items()}
        if _key(trial["input"]) not in seen:
            return trial
    raise AssertionError("finite uniqueness repair exhausted")


def _fallback_training(rng, scales, lengths, direction):
    """Finite constructive witnesses, reserved for exhausted evidence selection."""
    pairs = []
    for index, (k, p) in enumerate(zip(scales, lengths)):
        bg, frame, a, b, c, d = [int(v) for v in rng.permutation(10)[:6]]
        first, second = [a, b, d], [b, d, a]
        word = np.array({1: [first], 2: [first, second],
                         3: [first, second, first],
                         4: [first, second, second, second]}[p], dtype=np.int8)
        stencil = np.full((3, 5), bg, dtype=np.int8)
        for (r, col), ink in list(zip(((0, 0), (1, 2), (2, 4), (0, 3)), (c, a, c, b)))[:k]:
            stencil[r, col] = ink
        auxiliary = np.full((2, 3), bg, dtype=np.int8)
        auxiliary_k = 4 if (scales[0], lengths[1]) == (3, 2) else 3
        for (r, col), ink in list(zip(((0, 2), (1, 0), (1, 2), (0, 1)), (c, c, a, b)))[:auxiliary_k]:
            auxiliary[r, col] = ink
        heads = [dict(k=k, word=word, stencil=stencil, box=(1, 1, 5, 7), direction=0),
                 dict(k=auxiliary_k, word=np.array([[a, b], [b, a]], dtype=np.int8),
                      stencil=auxiliary, box=(8, 3, 4, 5), direction=0)]
        pair = _paint((14, 30), heads, bg, frame)
        pairs.append({role: np.ascontiguousarray(np.rot90(g, (direction + 2*index) % 4))
                      for role, g in pair.items()})
    assert _REQUIRED <= set().union(*(_evidence(pair) for pair in pairs))
    return pairs


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    scales = [int(x) for x in rng.choice([2, 3, 4], size=2, replace=False)]
    lengths = [3, int(rng.choice([2, 4]))]
    rng.shuffle(lengths)
    heldout = (scales[0], lengths[1])
    direction = int(rng.integers(4))
    # Evidence-selected witnesses; neither presentation index fixes a picture.
    train = None
    for _ in range(40):
        proposed = [_make_pair(rng, scales[i], lengths[i], heldout,
                               direction=(direction + 2*i) % 4, whole_word=True)
                    for i in range(2)]
        if (_key(proposed[0]["input"]) != _key(proposed[1]["input"])
                and _REQUIRED <= set().union(*(_evidence(pair) for pair in proposed))):
            train = proposed
            break
    if train is None:
        train = _fallback_training(rng, scales, lengths, direction)
    puzzle, seen = {"train": train, "test": []}, {_key(p["input"]) for p in train}
    options = [(a, b) for a in range(1, 5) for b in range(1, 5) if (a, b) != heldout]
    for split, total in (("train", num_train), ("test", num_test)):
        while len(puzzle[split]) < total:
            a, b = heldout if split == "test" else options[int(rng.integers(len(options)))]
            pair = None
            for _ in range(32 if split == "test" else 1):
                candidate = _make_pair(rng, a, b, heldout)
                if split == "train" or _test_evidence(candidate, heldout):
                    pair = candidate
                    break
            if pair is None:
                other_scale = next(c for c in (2, 3, 4) if c != a)
                other_length = 3 if b != 3 else 2
                pair = _fallback_training(rng, [a, other_scale], [b, other_length],
                                          int(rng.integers(4)))[0]
            pair = _unique_pair(pair, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _test_evidence(pair, combination=None):
    g, bg, _, heads = _analyze_input(pair["input"])
    if combination is not None:
        heads = [s for s in heads if (s["k"], len(s["word"])) == combination]
    expected = _render(g, bg, heads)
    return bool(heads) and all(not np.array_equal(_render(g, bg, heads, name), expected)
               for name in ("last_slice", "no_cycle", "plain_extension", "no_count", "ignore_stencil"))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses, evidence = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, bg, frame, heads = _analyze_input(pair["input"])
            out = _grid(pair["output"])
            assert out.shape == g.shape and _key(g) not in seen
            seen.add(_key(g))
            assert np.array_equal(_render(g, bg, heads), out), "wrong ribbon or damaged original"
            changed_region = set()
            for s in heads:
                p = len(s["word"])
                for d in range(p+1, s["maximum"]+1):
                    for q in _coords(s["box"], s["direction"], d):
                        changed_region.add(q)
                        assert g[q] == bg and out[q] not in (bg, frame)
            assert set(zip(*np.nonzero(g != out))) == changed_region
            analyses[split].extend(heads)
            if split == "train":
                evidence |= _evidence(pair)
    assert _REQUIRED <= evidence, "demonstrations leave a listed shortcut unresolved"
    training = {(s["k"], len(s["word"])) for s in analyses["train"]}
    ks, ps = {a for a, b in training}, {b for a, b in training}
    assert len(ks) >= 2 and len(ps) >= 2
    for pair in puzzle["test"]:
        heads = _analyze_input(pair["input"])[3]
        assert any((s["k"], len(s["word"])) not in training
                   and s["k"] in ks and len(s["word"]) in ps
                   and _test_evidence(pair, (s["k"], len(s["word"])))
                   for s in heads), "missing visible recombination shift"


def family_features(pair):
    """Public, input-derived measurements for local review and diversity checks."""
    g, bg, frame, heads = _analyze_input(pair["input"])
    return dict(shape=g.shape, background=bg, frame=frame, heads=len(heads),
                directions=tuple(s["direction"] for s in heads),
                counts=tuple(s["k"] for s in heads),
                lengths=tuple(len(s["word"]) for s in heads),
                widths=tuple(s["word"].shape[1] for s in heads),
                stencil_shapes=tuple(s["stencil"].shape for s in heads),
                extensions=tuple(s["maximum"]-len(s["word"]) for s in heads),
                clipped=any((s["maximum"]-len(s["word"])) % (s["k"]*len(s["word"])) for s in heads),
                words=tuple(_normal(s["word"]) for s in heads),
                stencils=tuple(_normal(s["stencil"]) for s in heads),
                combinations=tuple((s["k"], len(s["word"]), s["stencil"].shape,
                                    s["direction"], s["maximum"]-len(s["word"])) for s in heads))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fs = [family_features(pair) for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert len({_key(p["test"][0]["input"]) for p in puzzles}) == 42
    assert len({f["shape"] for f in fs}) >= 35
    assert {f["background"] for f in fs} == set(range(10))
    assert {f["frame"] for f in fs} == set(range(10))
    assert {v for f in fs for v in f["counts"]} == {1, 2, 3, 4}
    assert {v for f in fs for v in f["lengths"]} == {1, 2, 3, 4}
    assert {v for f in fs for v in f["widths"]} == {2, 3, 4}
    assert {v for f in fs for v in f["directions"]} == set(range(4))
    assert {f["heads"] for f in fs} == {1, 2, 3}
    assert len({v for f in fs for v in f["words"]}) >= 70
    assert len({v for f in fs for v in f["stencils"]}) >= 100
    assert len({v for f in fs for v in f["combinations"]}) >= 150
    assert any(f["clipped"] for f in fs)
    assert any(len(set(d//2 for d in f["directions"])) == 2 for f in fs)
    assert any(1 in f["extensions"] for f in fs)
    repeat = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for split in puzzles[0]
               for a, b in zip(puzzles[0][split], repeat[split]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(4200 + 10*a+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for index, (scales, lengths) in enumerate(itertools.product(
            itertools.permutations([2, 3, 4], 2),
            ((3, 1), (1, 3), (3, 2), (2, 3), (3, 4), (4, 3)))):
        fallback = _fallback_training(np.random.default_rng(index), scales, lengths, index % 4)
        for pair in fallback:
            assert np.array_equal(solve(pair["input"]), pair["output"])
            if lengths[0] > 1:
                assert _test_evidence(fallback[0], (scales[0], lengths[0]))
        heldout = (scales[0], lengths[1])
        assert all((s["k"], len(s["word"])) != heldout
                   for pair in fallback for s in _analyze_input(pair["input"])[3])
    # Recoloring repairs preserve semantic roles while guaranteeing uniqueness.
    repaired = _unique_pair(puzzles[0]["train"][0], {_key(puzzles[0]["train"][0]["input"])})
    assert np.array_equal(solve(repaired["input"]), repaired["output"])
    # This also checks that validation accepts compact nested-list serialization.
    damaged = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    validate_generated_puzzle(damaged)
    damaged["test"][0]["output"][0][0] = (damaged["test"][0]["output"][0][0] + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 diversity seeds, all legal counts, mutant rejection)")
