"""Inward-sampled radial stamps. No parent module or external data is used.

All generation and tests are in memory. This module writes no files or traces.
See task_spec.md for the visible geometry, including even-size sampling ties.
"""
from functools import lru_cache
import hashlib

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
HOST_SEEDS = (0, 161821341, 372843394, 2328978, 268278586, 417080057,
              360096003, 372875131, 641729721, 847568562, 730624397,
              82765821, 915330460, 532972638, 665791514, 902233579,
              68663407, 959554274, 921636107, 675706591, 893928263,
              858919021, 792021782, 435061804, 716752532, 288519508,
              10455993, 322952706, 993176880, 864273148, 891759509,
              6632709, 501441030, 134578674, 834078019, 287288257,
              260282195, 698500167, 956434347, 201458629, 430032201,
              249485464)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad color")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _views(a):
    return tuple(np.rot90(a, k) for k in range(4)) + tuple(
        np.rot90(a[:, ::-1], k) for k in range(4))


def _nearest(n, dr, dc):
    """All Euclidean-nearest cells to the overall ring center, within a tile."""
    def axis(d):
        if d:
            return (n - 1,) if d < 0 else (0,)
        return ((n // 2,) if n % 2 else (n // 2 - 1, n // 2))
    return tuple((r, c) for r in axis(dr) for c in axis(dc))


def _analyze(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    points = np.argwhere(a != bg)
    _require(bool(len(points)), "empty input")
    lo, hi = points.min(0), points.max(0) + 1
    r, c = map(int, lo)
    side, width = map(int, hi - lo)
    _require(side == width and side in (6, 9), "expected a 6x6 or 9x9 ring")
    n = side // 3
    _require(r >= 1 and c >= 1 and hi[0] < len(a) and hi[1] < a.shape[1],
             "ring needs a uniform border of at least one cell")
    occupied = np.zeros(a.shape, bool)
    occupied[r:r + side, c:c + side] = True
    occupied[r + n:r + 2 * n, c + n:c + 2 * n] = False
    _require(np.array_equal(occupied, a != bg), "noise, missing source cells, or filled hole")
    colors, counts = np.unique(a[occupied], return_counts=True)
    _require(len(colors) == 2 and counts[0] != counts[1], "need two roles and a strict majority")
    carrier = int(colors[np.argmax(counts)])
    ink = int(colors[np.argmin(counts)])
    glyph = np.full((3, 3), carrier, dtype=int)
    tiles, samples = [], []
    for dr, dc in DIRECTIONS:
        tr, tc = r + (dr + 1) * n, c + (dc + 1) * n
        tile = a[tr:tr + n, tc:tc + n].copy()
        values = {int(tile[p]) for p in _nearest(n, dr, dc)}
        _require(len(values) == 1, "tied nearest cells must have the same color")
        sample = values.pop()
        glyph[1 + dr, 1 + dc] = sample
        tiles.append(tile)
        samples.append(sample)
    _require(len(set(samples)) == 2, "both sample phases must be visible")
    return dict(n=n, background=bg, carrier=carrier, ink=ink, origin=(r, c),
                tiles=tuple(tiles), samples=tuple(samples), glyph=glyph)


def _render(info, behavior="normal", operation=2):
    """Block construction. Variants are finite competing hypotheses for coverage."""
    n, bg, carrier, ink = (info[k] for k in ("n", "background", "carrier", "ink"))
    samples = list(info["samples"])
    if behavior == "group_samples":
        samples = [samples[0] if dr and dc else samples[1] for dr, dc in DIRECTIONS]
    elif behavior == "majority_samples":
        samples = [carrier] * 8
    elif behavior == "outward_samples":
        samples = [int(t[_nearest(n, -dr, -dc)[0]])
                   for t, (dr, dc) in zip(info["tiles"], DIRECTIONS)]
    glyph = np.full((3, 3), carrier, int)
    for (dr, dc), color in zip(DIRECTIONS, samples):
        glyph[1 + dr, 1 + dc] = color
    exchanged = carrier + ink - glyph
    if behavior == "no_exchange":
        exchanged = glyph
    elif behavior == "rotate_stamp":
        exchanged = exchanged[::-1, ::-1]
    length = 3 * n
    out = np.full((3 * length, 3 * length), bg, int)
    out[length:2 * length, length:2 * length] = np.repeat(np.repeat(glyph, n, 0), n, 1)
    for tile, sample, (dr, dc) in zip(info["tiles"], samples, DIRECTIONS):
        phase = ink if behavior == "ink_gate" else carrier if behavior == "carrier_gate" else sample
        mask = tile == phase
        mask = _views(mask)[operation]
        if behavior == "fill_rectangles":
            mask = np.ones((n, n), bool)
        stamp = np.full((3, 3), sample, int) if behavior == "solid_sample" else exchanged
        block = np.where(np.repeat(np.repeat(mask, 3, 0), 3, 1),
                         np.tile(stamp, (n, n)), bg)
        top, left = (dr + 1) * length, (dc + 1) * length
        out[top:top + length, left:left + length] = block
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, sample, stencil and destination exclusively from input."""
    return _render(_analyze(input_grid))


@lru_cache(maxsize=2)
def _catalog(n):
    # At most 528 tiny masks across the two cache entries; no persistent storage.
    records = []
    for bits in range(1 << (n * n)):
        mask = np.array([(bits >> i) & 1 for i in range(n * n)], bool).reshape(n, n)
        count = int(mask.sum())
        asymmetric = len({v.tobytes() for v in _views(mask)}) == 8
        records.append((mask, count, asymmetric))
    return tuple(records)


def _glyph_bits(rng, count_range, witness):
    candidates = []
    for bits in range(1, 255):
        count = bits.bit_count()
        if count not in count_range:
            continue
        glyph = np.zeros((3, 3), bool)
        for i, (dr, dc) in enumerate(DIRECTIONS):
            glyph[1 + dr, 1 + dc] = bool((bits >> i) & 1)
        if not witness or len({v.tobytes() for v in _views(glyph)}) == 8:
            candidates.append(glyph)
    _require(bool(candidates), "empty glyph range")
    return candidates[int(rng.integers(len(candidates)))].copy()


def _make_tiles(rng, n, glyph, witness):
    options = []
    for dr, dc in DIRECTIONS:
        phase = bool(glyph[1 + dr, 1 + dc])
        near = _nearest(n, dr, dc)
        choices = [(mask, count) for mask, count, asym in _catalog(n)
                   if all(bool(mask[p]) == phase for p in near)
                   and (not witness or (asym and 2 <= count <= 4))]
        _require(bool(choices), "no compatible stencil")
        options.append(choices)
    # A global budget allows a dense/uniform individual tile, without losing the
    # visibly unique carrier. Random ordering avoids a favored compass direction.
    order = list(map(int, rng.permutation(8)))
    minimum = [min(count for _, count in opts) for opts in options]
    remaining = 4 * n * n - 1
    tiles = [None] * 8
    for step, index in enumerate(order):
        reserve = sum(minimum[j] for j in order[step + 1:])
        choices = [(mask, count) for mask, count in options[index] if count <= remaining - reserve]
        target = float(rng.uniform(.3, .8) * n * n)
        weights = np.array([np.exp(-.24 * (count - target) ** 2) for _, count in choices])
        mask, count = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        tiles[index] = mask.copy()
        remaining -= count
    return tiles


def _make_pair(rng, n, count_range, witness, margin_mode):
    glyph = _glyph_bits(rng, count_range, witness)
    tiles = _make_tiles(rng, n, glyph, witness and n == 3)
    bg, carrier, ink = map(int, rng.choice(10, 3, replace=False))
    side = 3 * n
    if margin_mode == "small":
        h, w = map(int, rng.integers(side + 2, side + 7, size=2))
    elif margin_mode == "large":
        h, w = map(int, rng.integers(side + 8, 31, size=2))
    else:
        h, w = map(int, rng.integers(side + 2, 31, size=2))
    top, left = int(rng.integers(1, h - side)), int(rng.integers(1, w - side))
    a = np.full((h, w), bg, int)
    for bits, (dr, dc) in zip(tiles, DIRECTIONS):
        r, c = top + (dr + 1) * n, left + (dc + 1) * n
        a[r:r + n, c:c + n] = np.where(bits, ink, carrier)
    return {"input": a, "output": solve(a)}


BEHAVIORS = ("no_exchange", "rotate_stamp", "fill_rectangles", "ink_gate",
             "carrier_gate", "group_samples", "majority_samples",
             "outward_samples", "solid_sample")


def _coverage(train):
    infos = [_analyze(p["input"]) for p in train]
    # Center completion and phase selection must both affect demonstrated answers.
    for behavior in BEHAVIORS:
        if not any(not np.array_equal(_render(info, behavior), p["output"])
                   for info, p in zip(infos, train)):
            return False
    # Eight D4 stencil interpretations per direction, including the review's
    # radial-reflection shortcut. This checks source masks, not generation flags.
    for index in range(8):
        survivors = set(range(8))
        for info in infos:
            mask = info["tiles"][index] == info["samples"][index]
            actual = mask[::-1, ::-1]
            survivors.intersection_update(k for k, view in enumerate(_views(mask))
                                          if np.array_equal(view, actual))
        if survivors != {2}:
            return False
    return {info["n"] for info in infos} == {2, 3}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    train_range, test_range = ((range(2, 5), range(5, 8)) if shift == 0 else
                               (range(4, 7), range(1, 4)) if shift == 1 else
                               (range(2, 7), range(1, 8)))
    train_margin, test_margin = ("small", "large") if shift == 2 else ("any", "any")
    # Coverage repair changes only training. Maximum 24 candidates, at most ten
    # retained pairs. Test labels never influence which conventions are taught.
    train = []
    for _ in range(24):
        train = [_make_pair(rng, 2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4)),
                            train_range, i < 2, train_margin) for i in range(num_train)]
        if _coverage(train) and len({_key(p["input"]) for p in train}) == num_train:
            break
    else:
        raise AssertionError("bounded training coverage construction failed")
    rng.shuffle(train)
    seen = {_key(p["input"]) for p in train}
    tests = []
    for _ in range(num_test):
        for _attempt in range(24):
            pair = _make_pair(rng, int(rng.integers(2, 4)), test_range, False, test_margin)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("bounded distinct-input construction failed")
        seen.add(_key(pair["input"]))
        tests.append(pair)
    return {"train": train, "test": tests}


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    info = _analyze(a)
    n, bg, carrier, ink = (info[k] for k in ("n", "background", "carrier", "ink"))
    length = 3 * n
    _require(b.shape == (9 * n, 9 * n), "wrong output size")
    # Scalar inverse coordinates independently check the slice-based painter.
    # The center is a visible enlargement; each outer sector is phase-selective.
    glyph = info["glyph"]
    selected = 0
    for y in range(9 * n):
        for x in range(9 * n):
            dr, dc = y // length - 1, x // length - 1
            ly, lx = y % length, x % length
            if dr == dc == 0:
                expected = int(glyph[ly // n, lx // n])
            else:
                index = DIRECTIONS.index((dr, dc))
                sr, sc = n - 1 - ly // 3, n - 1 - lx // 3
                active = info["tiles"][index][sr, sc] == info["samples"][index]
                expected = carrier + ink - int(glyph[ly % 3, lx % 3]) if active else bg
                selected += int(active)
            _require(int(b[y, x]) == expected, "incorrect hub, phase gate, stencil, or stamp")
    count = sum(int(np.count_nonzero(t == s)) for t, s in zip(info["tiles"], info["samples"]))
    _require(selected == 9 * count, "wrong expansion area")
    _require(int(np.count_nonzero(b != bg)) == length * length + 9 * count,
             "transparent cells or block support changed")
    _require(set(map(int, np.unique(b))) <= {bg, carrier, ink}, "unexpected output color")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad pair counts")
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
    _require(_coverage(puzzle["train"]), "demonstrations lack scale or competing-rule witnesses")


def self_test():
    """42 host seeds, one retained puzzle at a time, no files or execution traces."""
    signatures, layouts, glyphs, stencils = set(), set(), set(), set()
    backgrounds, carriers, inks, scales, sample_counts = set(), set(), set(), set(), set()
    uniform_tiles = dense_tiles = disconnected = 0
    split_scales = {"train": set(), "test": set()}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = np.asarray(pair["input"])
                b = np.asarray(pair["output"])
                info = _analyze(a)
                n = info["n"]
                digest.update(bytes(a.shape) + a.astype(np.uint8).tobytes())
                digest.update(bytes(b.shape) + b.astype(np.uint8).tobytes())
                layouts.add((a.shape, info["origin"]))
                split_scales[split].add(n)
                scales.add(n)
                backgrounds.add(info["background"])
                carriers.add(info["carrier"])
                inks.add(info["ink"])
                bits = info["glyph"] == info["ink"]
                glyphs.add(bits.tobytes())
                sample_counts.add(int(bits.sum()))
                for tile, sample in zip(info["tiles"], info["samples"]):
                    mask = tile == sample
                    stencils.add((n, mask.tobytes()))
                    uniform_tiles += int(mask.all())
                    dense_tiles += int(np.count_nonzero(tile == info["ink"]) > n * n / 2)
                    # Four-connected occupancy, measured rather than assumed.
                    points = set(map(tuple, np.argwhere(mask)))
                    components = 0
                    while points:
                        components += 1
                        todo = [points.pop()]
                        while todo:
                            y, x = todo.pop()
                            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                                if q in points:
                                    points.remove(q)
                                    todo.append(q)
                    disconnected += int(components > 1)
                _require(np.array_equal(solve(a), b), "solver disagreement")
        signatures.add(digest.digest())
    _require(len(signatures) == 42, "seed diversity collapsed")
    _require(scales == {2, 3} and all(v == {2, 3} for v in split_scales.values()), "missing scale")
    _require(backgrounds == carriers == inks == set(range(10)), "palette-role diversity collapsed")
    _require(len(layouts) >= 100 and len(glyphs) >= 70 and len(stencils) >= 200,
             "structural variation collapsed")
    _require(sample_counts == set(range(1, 8)), "sample density variation missing")
    _require(min(uniform_tiles, dense_tiles, disconnected) > 10, "motif extremes missing")
    # Determinism and every legal explicit request, without retaining all runs.
    def signature(p):
        return tuple((_key(q["input"]), _key(q["output"]))
                     for s in ("train", "test") for q in p[s])
    _require(signature(generate_puzzle(1234, 2, 1)) == signature(generate_puzzle(1234, 2, 1)),
             "nondeterministic generation")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "ignored counts")
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Corrupt three semantically distinct regions and check rejection.
    p = generate_puzzle(0, 2, 1)
    original = np.asarray(p["test"][0]["output"]).copy()
    n = _analyze(p["test"][0]["input"])["n"]
    for y, x in ((0, 0), (3 * n, 3 * n), (9 * n - 1, 9 * n - 1)):
        broken = original.copy()
        broken[y, x] = (int(broken[y, x]) + 1) % 10
        p["test"][0]["output"] = broken
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "layouts": len(layouts),
            "glyphs": len(glyphs), "stencils": len(stencils),
            "uniform_tiles": uniform_tiles, "dense_tiles": dense_tiles,
            "disconnected_stencils": disconnected}
