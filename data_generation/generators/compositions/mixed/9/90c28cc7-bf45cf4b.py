"""Recover a run-enlarged binary mask, then substitute an unchanged stamp.

Generation writes no files. All solving conventions are visible in one input.
Only the standard library and NumPy are used; see task_spec.md for the domain.
"""
from __future__ import annotations

import hashlib
import numpy as np


VERIFICATION_SEEDS = (0, 878752461, 538210991, 97054293, 315380934,
    432918563, 985118429, 358131924, 14027748, 619389885, 441981672,
    570013468, 1963907, 376087814, 520817078, 65244684, 952996093,
    647557981, 866103737, 343834658, 351950538, 102315271, 905642222,
    164877775, 739331207, 359456225, 15195393, 581891967, 51517768,
    618221057, 769526328, 103101896, 645377676, 183470323, 975151794,
    516688457, 108979210, 733346568, 570017882, 379106701, 772679059,
    771122344)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _collapse(a):
    """Only adjacent *complete* vectors merge; separated recurrences survive."""
    ri = np.r_[0, 1 + np.flatnonzero(np.any(a[1:] != a[:-1], axis=1))]
    ci = np.r_[0, 1 + np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0))]
    return (a[np.ix_(ri, ci)].copy(),
            np.diff(np.r_[ri, a.shape[0]]), np.diff(np.r_[ci, a.shape[1]]))


def _tight(a):
    return all(bool(np.any(s)) for s in (a[0], a[-1], a[:, 0], a[:, -1]))


def _canonical(mask):
    return (not np.any(np.all(mask[1:] == mask[:-1], axis=1))
            and not np.any(np.all(mask[:, 1:] == mask[:, :-1], axis=0)))


def _blank_line(mask):
    return bool(np.any(~mask.any(axis=0)) or np.any(~mask.any(axis=1)))


def _recurrence(mask):
    return any(np.array_equal(a[i], a[j]) for a in (mask, mask.T)
               for i in range(len(a)) for j in range(i + 2, len(a)))


def _seam(mask):
    # A boundary hidden locally, witnessed by exactly one cell elsewhere.
    return any(np.count_nonzero(a[i] != a[i-1]) == 1 and a.shape[1] > 1
               for a in (mask, mask.T) for i in range(1, len(a)))


def _transforms(a):
    return [f(np.rot90(a, k)) for k in range(4)
            for f in (lambda x: x, np.fliplr)]


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in _transforms(a)[1:])


def _bbox(binary):
    rr, cc = np.nonzero(binary)
    if not len(rr):
        raise ValueError("empty encoding")
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _crop(a, b):
    return a[b[0]:b[1], b[2]:b[3]].copy()


def _separated(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _read(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not all(np.all(s == bg) for s in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("a uniform background rim is required")
    colors = set(map(int, np.unique(grid))) - {bg}
    candidates = []
    for signal in sorted(colors):
        if len(colors - {signal}) < 2:
            continue
        mb = _bbox(grid == signal)
        sb = _bbox((grid != bg) & (grid != signal))
        if not _separated(mb, sb):
            continue
        enlarged, stamp = _crop(grid, mb), _crop(grid, sb)
        if not np.all((enlarged == bg) | (enlarged == signal)) or signal in stamp:
            continue
        # An internal empty line in a stamp would permit competing object crops.
        if _blank_line(stamp != bg):
            continue
        mask, rows, cols = _collapse(enlarged == signal)
        if mask.shape[0] * stamp.shape[0] > 30 or mask.shape[1] * stamp.shape[1] > 30:
            continue
        candidates.append(dict(background=bg, signal=signal, mask=mask,
            stamp=stamp, rows=rows, columns=cols, enlarged=enlarged,
            mask_box=mb, stamp_box=sb))
    if len(candidates) != 1:
        raise ValueError("input must have one monochrome mask and one multicolor stamp")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer both roles, collapse only the mask, and copy the exact stamp."""
    d = _read(input_grid)
    mask, stamp, bg = d["mask"], d["stamp"], d["background"]
    sh, sw = stamp.shape
    rr, cc = np.indices((mask.shape[0] * sh, mask.shape[1] * sw))
    return np.where(mask[rr // sh, cc // sw], stamp[rr % sh, cc % sw], bg).astype(np.int8)


def _write_blocks(mask, stamp, bg):
    # Ground truth uses explicit block writes, separately from solve's indexing.
    h, w = stamp.shape
    out = np.full((mask.shape[0] * h, mask.shape[1] * w), bg, dtype=np.int8)
    for r, c in np.argwhere(mask):
        out[r*h:(r+1)*h, c*w:(c+1)*w] = stamp
    return out


def _shapes(rng, mode, test, witness, previous, tall, recur=False):
    ml, mu, sl, su = 2, 6, 2, 6
    if mode == "mask_growth":
        mu, su = (6 if test else 4), 4
    elif mode == "stamp_growth":
        mu, su = 5, (6 if test else 3)
    if witness:
        ml = sl = 3
    elif recur:
        ml = 3
    options = []
    for mh in range(ml, mu + 1):
        for mw in range(ml, mu + 1):
            if mode == "mask_growth" and test and max(mh, mw) < 5:
                continue
            if previous and (mh, mw) == previous[0]:
                continue
            for sh in range(sl, su + 1):
                for sw in range(sl, su + 1):
                    if mh * sh > 30 or mw * sw > 30:
                        continue
                    if mode == "stamp_growth" and test and max(sh, sw) < 4:
                        continue
                    if previous and (sh, sw) == previous[1]:
                        continue
                    if mode == "aspect":
                        want_tall = tall != test
                        if not (mh*sh > mw*sw if want_tall else mh*sh < mw*sw):
                            continue
                    options.append(((mh, mw), (sh, sw)))
    if not options:
        raise RuntimeError("no feasible mask/stamp shapes")
    return options[int(rng.integers(len(options)))]


def _mask(rng, shape, witness=False, recur=False, allow_blank=True):
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(1024):
        family = "scatter" if witness or recur else str(rng.choice(
            ["scatter", "scatter", "scatter", "checker", "frame", "diagonal", "steps"]))
        p = float(rng.choice([.22, .38, .5, .65, .8]))
        a = rng.random(shape) < p
        if family == "checker":
            a = (rr + cc) % 2 == int(rng.integers(2))
        elif family == "frame":
            a[0] = a[-1] = True
            a[:, 0] = a[:, -1] = True
        elif family == "diagonal" and h == w:
            a = np.eye(h, dtype=bool)
            if rng.random() < .5:
                a |= np.fliplr(a)
        elif family == "steps":
            a = rr * w >= cc * h
        if recur:
            # Copy a vector at a separated position, never erase its recurrence.
            if rng.random() < .5:
                a[-1] = a[0]
            else:
                a[:, -1] = a[:, 0]
        if rng.random() < .5:
            a = a[::-1].copy()
        if rng.random() < .5:
            a = a[:, ::-1].copy()
        if not _tight(a) or not _canonical(a) or not np.any(~a):
            continue
        if not allow_blank and _blank_line(a):
            continue
        if witness and (not _seam(a) or not _asymmetric(a)):
            continue
        if recur and not _recurrence(a):
            continue
        return a
    raise RuntimeError("could not sample a canonical mask")


def _scales(rng, shape, strong=False, witness=False, limit=4):
    for _ in range(256):
        profile = int(rng.integers(4))
        rows = rng.integers(1, limit + 1, size=shape[0])
        cols = rng.integers(1, limit + 1, size=shape[1])
        if not witness:
            if profile in (0, 1):
                rows[:] = int(rng.integers(1, limit + 1))
            if profile in (0, 2):
                cols[:] = int(rng.integers(1, limit + 1))
        if strong:
            axis = rows if rng.random() < .5 else cols
            axis[int(rng.integers(len(axis)))] = 5
        if max(rows) == max(cols) == 1 or rows.sum() > 20 or cols.sum() > 20:
            continue
        if witness and (len(set(rows)) < 2 or len(set(cols)) < 2):
            continue
        return rows, cols
    raise RuntimeError("could not fit expansion bands")


def _stamp(rng, shape, bg, payload, witness=False, repeat=False):
    rr, cc = np.indices(shape)
    n = len(payload)
    for _ in range(256):
        mode = 0 if witness else int(rng.integers(5))
        if mode == 0:
            labels = rng.integers(n, size=shape)
        elif mode == 1:
            labels = rr if rng.random() < .5 else cc
        elif mode == 2:
            labels = np.minimum.reduce([rr, cc, shape[0]-1-rr, shape[1]-1-cc])
        elif mode == 3:
            labels = rr + cc
        else:
            labels = abs(2*rr-shape[0]+1) + abs(2*cc-shape[1]+1)
        stamp = payload[labels % n].copy()
        if witness:
            stamp[1, 1] = bg
            boundary = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                        if r in (0, shape[0]-1) or c in (0, shape[1]-1)]
            stamp[boundary[int(rng.integers(len(boundary)))]] = bg
        elif rng.random() < .55:
            holes = rng.random(shape) < float(rng.uniform(.06, .3))
            stamp[holes] = bg
        if repeat:
            if rng.random() < .5:
                stamp[1] = stamp[0]
            else:
                stamp[:, 1] = stamp[:, 0]
        if len(set(map(int, np.unique(stamp))) - {bg}) < 2:
            continue
        if not _tight(stamp != bg) or _blank_line(stamp != bg):
            continue
        if witness and not _asymmetric(stamp):
            continue
        return stamp
    raise RuntimeError("could not construct a readable stamp")


def _embed(rng, enlarged, stamp, bg):
    mh, mw = enlarged.shape
    sh, sw = stamp.shape
    horizontal = bool(rng.integers(2))
    reverse = bool(rng.integers(2))
    gapmax = 28 - (mw + sw if horizontal else mh + sh)
    gap = int(rng.integers(1, min(6, gapmax) + 1))
    bh = (max(mh, sh) if horizontal else mh + sh + gap) + 2
    bw = (mw + sw + gap if horizontal else max(mw, sw)) + 2
    h = int(rng.integers(bh, min(30, bh + 8) + 1))
    w = int(rng.integers(bw, min(30, bw + 8) + 1))
    if horizontal:
        start = int(rng.integers(1, w - mw - sw - gap))
        mc, sc = (start, start + mw + gap) if not reverse else (start + sw + gap, start)
        mr, sr = int(rng.integers(1, h-mh)), int(rng.integers(1, h-sh))
    else:
        start = int(rng.integers(1, h - mh - sh - gap))
        mr, sr = (start, start + mh + gap) if not reverse else (start + sh + gap, start)
        mc, sc = int(rng.integers(1, w-mw)), int(rng.integers(1, w-sw))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[mr:mr+mh, mc:mc+mw] = enlarged
    grid[sr:sr+sh, sc:sc+sw] = stamp
    return grid


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("counts must satisfy 2<=train<=10, 1<=test<=4, train>test")
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(["mask_growth", "stamp_growth", "scale", "aspect"]))
    tall = bool(rng.integers(2))
    stable = bool(rng.random() < .25)
    shared = rng.permutation(10)
    puzzle, seen = {"train": [], "test": []}, set()
    previous = None
    train_blank = False
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            test = split == "test"
            witness, repeat = not test and index == 0, not test and index == 1
            for _ in range(32):
                mshape, sshape = _shapes(rng, mode, test, witness,
                                         previous if repeat else None, tall, repeat)
                mask = _mask(rng, mshape, witness, repeat, not test or train_blank)
                palette = shared if stable else rng.permutation(10)
                bg, signal = map(int, palette[:2])
                k = int(rng.integers(2, min(6, np.prod(sshape)) + 1))
                stamp = _stamp(rng, sshape, bg, palette[2:2+k], witness, repeat)
                rows, cols = _scales(rng, mshape, mode == "scale" and test,
                                     witness, 3 if mode == "scale" and not test else 4)
                enlarged = np.repeat(np.repeat(np.where(mask, signal, bg).astype(np.int8),
                                                rows, axis=0), cols, axis=1)
                grid = _embed(rng, enlarged, stamp, bg)
                if _key(grid) in seen:
                    continue
                info = _read(grid)
                assert np.array_equal(info["mask"], mask)
                assert np.array_equal(info["stamp"], stamp)
                seen.add(_key(grid))
                puzzle[split].append({"input": grid, "output": _write_blocks(mask, stamp, bg)})
                if not test:
                    previous = mshape, sshape
                    train_blank |= _blank_line(mask)
                break
            else:
                raise RuntimeError("could not produce distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _edge_holes(d):
    a = d["stamp"] == d["background"]
    return bool(np.any(a[0]) or np.any(a[-1]) or np.any(a[:, 0]) or np.any(a[:, -1]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            src, dst = _grid(pair["input"]), _grid(pair["output"])
            assert _key(src) not in seen
            seen.add(_key(src))
            d = _read(src)
            m, s, bg = d["mask"], d["stamp"], d["background"]
            assert _canonical(m) and _tight(m) and np.any(m) and np.any(~m)
            encoded = np.repeat(np.repeat(np.where(m, d["signal"], bg),
                                         d["rows"], axis=0), d["columns"], axis=1)
            assert np.array_equal(encoded, d["enlarged"])
            assert dst.shape == (m.shape[0]*s.shape[0], m.shape[1]*s.shape[1])
            assert d["signal"] not in dst
            sh, sw = s.shape
            for r in range(m.shape[0]):
                for c in range(m.shape[1]):
                    block = dst[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                    assert np.array_equal(block, s) if m[r, c] else np.all(block == bg)
            assert np.array_equal(solve(src), dst)
            # Omitting A uses enlarged pixels as placement cells: dimensions differ.
            assert max(d["rows"]) > 1 or max(d["columns"]) > 1
            assert d["enlarged"].shape != m.shape
            # Omitting B returns only a two-color logical mask, never this output.
            assert dst.shape != m.shape and len(set(np.unique(s)) - {bg}) >= 2
            infos[split].append(d)
    train, test = infos["train"], infos["test"]
    assert len({d["mask"].shape for d in train}) >= 2
    assert len({d["stamp"].shape for d in train}) >= 2
    assert any(_seam(d["mask"]) and _asymmetric(d["mask"])
               and len(set(d["rows"])) > 1 and len(set(d["columns"])) > 1 for d in train)
    assert any(_recurrence(d["mask"]) for d in train)
    assert any(_asymmetric(d["stamp"]) and _edge_holes(d)
               and np.any(d["stamp"][1:-1, 1:-1] == d["background"]) for d in train)
    assert any(not _canonical(d["stamp"]) for d in train), "show that stamps are not collapsed"
    if any(_blank_line(d["mask"]) for d in test):
        assert any(_blank_line(d["mask"]) for d in train)
    # At least one visible controlled shift, with no hidden test-time convention.
    def maximum(ds, role):
        return max(max(d[role].shape) for d in ds)
    grow_m = all(max(d["mask"].shape) > maximum(train, "mask") for d in test)
    grow_s = all(max(d["stamp"].shape) > maximum(train, "stamp") for d in test)
    bandmax = lambda d: max(max(d["rows"]), max(d["columns"]))
    grow_band = all(bandmax(d) > max(map(bandmax, train)) for d in test)
    aspect = lambda d: np.sign(d["mask"].shape[0]*d["stamp"].shape[0]
                               - d["mask"].shape[1]*d["stamp"].shape[1])
    flip_aspect = (len({aspect(d) for d in train}) == 1
                   and all(aspect(d) == -aspect(train[0]) != 0 for d in test))
    assert grow_m or grow_s or grow_band or flip_aspect


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _normalized(a, bg):
    roles = {bg: 0}
    return a.shape, bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)


def self_test():
    """42 supplied seeds, all 30 legal counts, and bounded semantic regressions.

    Puzzles are processed sequentially with small feature sets; no disk output.
    Returns compactly serializable descriptive measurements, not external approval.
    """
    masks, stamps, inputs, outputs, bands, palettes, layouts, puzzles = (set() for _ in range(8))
    flags = dict(recurrence=0, hidden_seam=0, stamp_holes=0, edge_holes=0,
                 repeated_stamp=0, blank_mask_line=0, rectangular_stamp=0)
    for seed in VERIFICATION_SEEDS:
        p = generate_puzzle(seed)
        puzzles.add(_digest(p))
        for split in ("train", "test"):
            for pair in p[split]:
                d = _read(pair["input"])
                m, s, bg = d["mask"], d["stamp"], d["background"]
                masks.add(_key(m)); stamps.add(_normalized(s, bg))
                inputs.add(pair["input"].shape); outputs.add(pair["output"].shape)
                bands.add((tuple(d["rows"]), tuple(d["columns"])))
                palettes.add((bg, d["signal"]))
                a, b = d["mask_box"], d["stamp_box"]
                layouts.add("left" if a[3] < b[2] else "right" if b[3] < a[2]
                            else "above" if a[1] < b[0] else "below")
                flags["recurrence"] += int(_recurrence(m))
                flags["hidden_seam"] += int(_seam(m))
                flags["stamp_holes"] += int(np.any(s == bg))
                flags["edge_holes"] += int(_edge_holes(d))
                flags["repeated_stamp"] += int(not _canonical(s))
                flags["blank_mask_line"] += int(_blank_line(m))
                flags["rectangular_stamp"] += int(s.shape[0] != s.shape[1])
    assert len(puzzles) == 42 and len(masks) >= 60 and len(stamps) >= 100
    assert len(inputs) >= 50 and len(outputs) >= 40 and len(bands) >= 100
    assert len(palettes) >= 30 and len(layouts) == 4 and all(flags.values())
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            p = generate_puzzle(0, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    # A literal recurrence fixture catches global deduplication and uneven bands.
    m = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype=bool)
    s = np.array([[2, 3, 3], [2, 3, 3], [0, 2, 3]], dtype=np.int8)
    enlarged = np.repeat(np.repeat(m.astype(np.int8)*7, [2, 1, 3], axis=0), [1, 3, 2], axis=1)
    canvas = np.zeros((11, 15), dtype=np.int8)
    canvas[1:7, 1:7] = enlarged
    canvas[6:9, 10:13] = s
    expected = np.array([[2,3,3,0,0,0,2,3,3], [2,3,3,0,0,0,2,3,3],
        [0,2,3,0,0,0,0,2,3], [0,0,0,2,3,3,2,3,3], [0,0,0,2,3,3,2,3,3],
        [0,0,0,0,2,3,0,2,3], [2,3,3,0,0,0,2,3,3], [2,3,3,0,0,0,2,3,3],
        [0,2,3,0,0,0,0,2,3]], dtype=np.int8)
    assert np.array_equal(solve(canvas), expected)
    assert _collapse(np.ones((4, 7), dtype=bool))[0].shape == (1, 1)
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=210, legal_count_combinations=30,
        distinct_puzzles=len(puzzles), mask_patterns=len(masks), stamp_patterns=len(stamps),
        input_shapes=len(inputs), output_shapes=len(outputs), expansion_profiles=len(bands),
        color_role_assignments=len(palettes), layouts=sorted(layouts), features=flags)
