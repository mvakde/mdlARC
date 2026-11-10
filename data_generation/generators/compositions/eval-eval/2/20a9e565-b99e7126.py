"""Growing self-portraits: extrapolation coupled to tile-scale self-similarity.

Only standard library and NumPy are used. No files, seeds or demonstrations are
consulted by solve. Generation paints latent stages; inference recovers stages
and fits their progression from the returned pixels.
"""
from collections import Counter
import hashlib

import numpy as np


STYLES = ("solid", "cap", "comb", "rails", "zig")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _checked_int(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def _corners(bb):
    a, b, c, d = bb
    return {(a, c), (a, c+1), (a+1, c), (b, d), (b, d-1), (b-1, d)}


def _find_corners(a):
    found = []
    for color in np.unique(a):
        points = np.argwhere(a == color)
        if len(points) != 6:
            continue
        r0, c0 = map(int, points.min(axis=0))
        r1, c1 = map(int, points.max(axis=0))
        if min(r1-r0, c1-c0) < 1 or r1-r0+c1-c0 < 4:
            continue
        bb = r0, r1, c0, c1
        if set(map(tuple, points)) == _corners(bb):
            found.append((int(color), bb))
    assert len(found) == 1, "exactly two canonical, disconnected corner Ls required"
    return found[0]


def _runs(indices):
    result = []
    for x in map(int, indices):
        if result and x == result[-1][-1]+1:
            result[-1].append(x)
        else:
            result.append([x])
    return result


def _lattice(a):
    d = int(a[0, 0])
    rr = _runs(np.flatnonzero(np.all(a == d, axis=1)))
    cc = _runs(np.flatnonzero(np.all(a == d, axis=0)))
    assert len(rr) >= 3 and len(cc) >= 3
    assert rr[0][0] == cc[0][0] == 0
    assert rr[-1][-1] == a.shape[0]-1 and cc[-1][-1] == a.shape[1]-1
    sizes = {len(v) for v in rr+cc}
    assert len(sizes) == 1
    t = sizes.pop()
    hh = {rr[i+1][0]-rr[i][-1]-1 for i in range(len(rr)-1)}
    ww = {cc[i+1][0]-cc[i][-1]-1 for i in range(len(cc)-1)}
    assert len(hh) == len(ww) == 1
    h, w = hh.pop(), ww.pop()
    assert 2 <= h <= 4 and 2 <= w <= 4 and t in (1, 2)
    return h, w, t, len(rr)-1, len(cc)-1, d


def _shape(style, h, w, fr=False, fc=False):
    """Inference representation: cell predicates, independent of stroke drawing."""
    r, c = np.indices((h, w))
    if style == "solid":
        mask = np.ones((h, w), dtype=bool)
    elif style == "cap":
        mask = (r == 0) | (c == 0)
    elif style == "comb":
        mask = (c == 0) | (r % 2 == 0)
    elif style == "rails":
        mask = (r == 0) | (r == h-1)
    elif style == "zig":
        mask = (r == 0) | (c == r % w)
    else:
        raise AssertionError("unknown stroke rule")
    return mask[::-1 if fr else 1, ::-1 if fc else 1].copy()


def _draw_shape(style, h, w):
    """Generation representation: explicit strokes, not an inference call."""
    a = np.zeros((h, w), dtype=bool)
    if style == "solid":
        a[:] = True
    elif style == "cap":
        a[0, :] = True
        a[:, 0] = True
    elif style == "comb":
        a[:, 0] = True
        for r in range(0, h, 2):
            a[r, :] = True
    elif style == "rails":
        a[0, :] = a[-1, :] = True
    elif style == "zig":
        a[0, :] = True
        for r in range(1, h):
            a[r, r % w] = True
    else:
        raise AssertionError("unknown stroke rule")
    return a


def _blank(geo, ordinary):
    h, w, t, nr, nc, d = geo
    a = np.full((nr*(h+t)+t, nc*(w+t)+t), d, dtype=np.int8)
    for r in range(nr):
        for c in range(nc):
            y, x = t+r*(h+t), t+c*(w+t)
            a[y:y+h, x:x+w] = ordinary
    return a


def _stamp(a, geo, r, c, tile):
    h, w, t, nr, nc, _ = geo
    if 0 <= r < nr and 0 <= c < nc:
        y, x = t+r*(h+t), t+c*(w+t)
        a[y:y+h, x:x+w] = tile


def _linear(values):
    first, step = int(values[0]), int(values[1])-int(values[0])
    assert all(int(v) == first+i*step for i, v in enumerate(values)), "non-arithmetic geometry"
    return first, step


def _crop(a, bb):
    r0, r1, c0, c1 = bb
    return a[r0:r1+1, c0:c1+1].copy()


def _predict(model, style, fr, fc, *, freeze_geometry=False, freeze_micro=False,
             solid_macro=False, micro_transform=None, ink_only=False):
    geo, bb, ordinary = model["geo"], model["bb"], model["ordinary"]
    h, w, t, nr, nc, _ = geo
    answer = _blank(geo, ordinary)
    p0, p1 = (bb[2]-t)//(w+t), (bb[3]-t)//(w+t)
    n = model["n"]
    right = model["right"]
    near = n-1 if right else 0
    targets, new_tiles = [], []
    for i in range(-30, 31):
        x = model["x"][0]+i*model["x"][1]
        sw = model["sw"]
        if x > p1 or x+sw-1 < p0:
            continue
        if 0 <= i < n:
            continue
        assert (i >= n) == right, "crop crosses the observed sequence"
        j = near if freeze_geometry else i
        sh = model["height"][0]+j*model["height"][1]
        y = model["y"][0]+i*model["y"][1]
        ar = model["ar"][0]+j*model["ar"][1]
        ac = model["ac"][0]+j*model["ac"][1]
        assert 1 <= sh <= h and 0 <= ar <= h-sh and 0 <= ac <= w-sw
        mask = _shape(style, sh, sw, fr, fc)
        fine_mask = np.zeros((h, w), dtype=bool)
        fine_mask[ar:ar+sh, ac:ac+sw] = mask
        if freeze_micro:
            fine_mask = model["fine_masks"][near]
        if micro_transform is not None:
            fine_mask = micro_transform(fine_mask)
        ink = model["cycle"][i % len(model["cycle"])]
        tile = np.where(fine_mask, ink, model["background"]).astype(np.int8)
        positions = np.argwhere(np.ones_like(mask) if solid_macro else mask)
        hits = []
        for r, c in positions:
            r, c = y+int(r), x+int(c)
            py, px = t+r*(h+t), t+c*(w+t)
            if (0 <= r < nr and 0 <= c < nc and py <= bb[1] and py+h-1 >= bb[0]
                    and px <= bb[3] and px+w-1 >= bb[2]):
                painted = np.where(fine_mask, ink, ordinary) if ink_only else tile
                _stamp(answer, geo, r, c, painted)
                hits.append((r, c))
        if hits:
            targets.append(i)
            new_tiles.extend(hits)
    assert targets, "crop must intersect an active future tile"
    return _crop(answer, bb), {"target_indices": tuple(targets), "target_tiles": tuple(new_tiles)}


def _canonical(a, marker, bb):
    geo = _lattice(a)
    h, w, t, nr, nc, d = geo
    records, patches = Counter(), {}
    for r in range(nr):
        for c in range(nc):
            y, x = t+r*(h+t), t+c*(w+t)
            tile = a[y:y+h, x:x+w]
            if marker not in tile:
                key = _key(tile)
                records[key] += 1
                patches[key] = tile.copy()
    ordered = records.most_common()
    assert len(ordered) >= 2 and ordered[0][1] > ordered[1][1], "unique ordinary tile required"
    ordinary = patches[ordered[0][0]]
    ordinary_colors = set(map(int, np.unique(ordinary)))
    assert len(ordinary_colors) in (2, 3) and marker not in ordinary_colors and d not in ordinary_colors
    clean = a.copy()
    labels, glyphs = {}, {}
    for r in range(nr):
        for c in range(nc):
            y, x = t+r*(h+t), t+c*(w+t)
            tile = a[y:y+h, x:x+w].copy()
            annotated = tile == marker
            if annotated.any():
                tile[annotated] = ordinary[annotated]
                clean[y:y+h, x:x+w] = tile
            assert d not in tile
            if np.array_equal(tile, ordinary):
                continue
            assert not annotated.any(), "markers must lie in the unobserved continuation"
            key = _key(tile)
            labels[r, c] = key
            glyphs[key] = tile
    # Every occupied coarse column belongs to one stage. Equal neighboring
    # stages need an ordinary gap; alternating ink may separate touching stages.
    groups, previous = [], None
    for c in range(nc):
        keys = {v for (r, x), v in labels.items() if x == c}
        assert len(keys) <= 1, "stages overlap along their progression axis"
        current = next(iter(keys)) if keys else None
        if current is not None:
            if current == previous:
                groups[-1][1].append(c)
            else:
                groups.append((current, [c]))
        previous = current
    assert 3 <= len(groups) <= 5, "at least three complete stages required"
    stages = []
    for key, columns in groups:
        tile = glyphs[key]
        extra = set(map(int, np.unique(tile)))-ordinary_colors
        assert len(extra) == 1 and marker not in extra and d not in extra
        ink = extra.pop()
        mask = tile == ink
        assert np.unique(tile[~mask]).size == 1
        background = int(tile[~mask][0])
        assert background in ordinary_colors
        pts = np.argwhere(mask)
        ar, ac = map(int, pts.min(axis=0))
        br, bc = map(int, pts.max(axis=0))
        small = mask[ar:br+1, ac:bc+1]
        assert np.all(small.any(axis=0))
        actual = {(r, c) for (r, c), v in labels.items() if v == key and c in columns}
        y, x = min(r for r, c in actual), min(c for r, c in actual)
        relative = {(r-y, c-x) for r, c in actual}
        assert relative == set(map(tuple, np.argwhere(small))), "micro and macro silhouettes disagree"
        stages.append(dict(height=br-ar+1, sw=bc-ac+1, ar=ar, ac=ac, y=y, x=x,
                           small=small, fine=mask, ink=ink, background=background))
    assert len({v["background"] for v in stages}) == 1
    assert len({v["sw"] for v in stages}) == 1
    model = {name: _linear([s[name] for s in stages]) for name in ("height", "ar", "ac", "y", "x")}
    assert model["x"][1] >= stages[0]["sw"]
    assert abs(model["height"][1]) <= 1
    colors = [s["ink"] for s in stages]
    cycles = [tuple(colors[:p]) for p in (1, 2) if all(v == colors[i % p] for i, v in enumerate(colors))]
    assert cycles, "ink must be constant or alternate with a witnessed return"
    last_pixel = t+(stages[-1]["x"]+stages[-1]["sw"]-1)*(w+t)+w-1
    first_pixel = t+stages[0]["x"]*(w+t)
    assert bb[2] > last_pixel or bb[3] < first_pixel
    right = bb[2] > last_pixel
    model.update(geo=geo, ordinary=ordinary, bb=bb, background=stages[0]["background"],
                 cycle=cycles[0], sw=stages[0]["sw"], n=len(stages), right=right,
                 fine_masks=[s["fine"] for s in stages])
    # Ensure all annotation cells are tile interiors, so divider detection is
    # lossless, and the entire requested region contains only the ordinary field.
    for r, c in _corners(bb):
        assert (r-t) % (h+t) < h and (c-t) % (w+t) < w
        assert r >= t and c >= t
    assert np.array_equal(_crop(clean, bb), _crop(_blank(geo, ordinary), bb))
    answers = []
    for style in STYLES:
        for fr in (False, True):
            for fc in (False, True):
                if not all(np.array_equal(s["small"], _shape(style, s["height"], s["sw"], fr, fc)) for s in stages):
                    continue
                try:
                    answer, info = _predict(model, style, fr, fc)
                except AssertionError:
                    continue
                answers.append((answer, model, style, fr, fc, info))
    assert answers, "no supported evolving self-portrait"
    return answers


def _analyze(input_grid):
    a = np.asarray(input_grid)
    assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
    assert all(1 <= n <= 30 for n in a.shape) and np.all((a >= 0) & (a <= 9))
    marker, bb = _find_corners(a)
    answers = []
    for transpose in (False, True):
        work = a.T if transpose else a
        bounds = (bb[2], bb[3], bb[0], bb[1]) if transpose else bb
        try:
            candidates = _canonical(work, marker, bounds)
        except AssertionError:
            continue
        for answer, model, style, fr, fc, info in candidates:
            answer = answer.T.copy() if transpose else answer
            answers.append((answer, model, style, fr, fc, info, transpose))
    assert answers, "input does not determine a valid continuation"
    answer = answers[0][0]
    assert all(np.array_equal(answer, v[0]) for v in answers), "competing models disagree on crop"
    _, m, style, fr, fc, extra, transpose = answers[0]
    info = dict(model=m, style=style, fr=fr, fc=fc, transpose=transpose, marker=marker,
                bounds=bb, candidates=len(answers), **extra)
    return answer, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the inclusive requested crop, using only the input pixels."""
    return _analyze(input_grid)[0]


def _ablation(info, **options):
    result, _ = _predict(info["model"], info["style"], info["fr"], info["fc"], **options)
    return result.T.copy() if info["transpose"] else result


def _construct(rng, *, test, index, axis, fallback=None):
    if fallback is not None:
        th, tw, t, nr, nc = 4, 2, 1, 5, 9
        sh, dh = (4, -1) if not test and index == 1 else (1, 1)
        style, sw, n, gap = "comb", 2, 3, 0
        y0, dy, x0, skip, span = 0, 0, 0, 0, 1
        roles = [0, 1, 2, 3, 4, 5]
        ordinary = np.array([roles[1+(fallback >> i & 1)] for i in range(8)], dtype=np.int8).reshape(4, 2)
        colors, bg, marker, divider = (3, 4), 1, 5, 0
        if not test and index == 0:
            # The next comb's non-ink cells must witness whole-tile copying.
            bg = 2 if np.all(ordinary[[1, 3], 1] == 1) else 1
        anchor, flip_r, flip_c, reverse = 0, False, False, False
    else:
        mandatory_growth = not test and index < 2
        dh = (1 if index == 0 else -1) if mandatory_growth else int(rng.choice([-1, 0, 0, 1]))
        sh = (1 if dh > 0 else 4) if dh else int(rng.integers(2, 5))
        th = 4 if dh else int(rng.integers(sh, 5))
        sw = 2 if mandatory_growth else int(rng.choice([1, 2], p=[.35, .65]))
        tw = int(rng.integers(max(2, sw), 5))
        t = int(rng.choice([1, 2], p=[.85, .15]))
        nr, nc = (30-t)//(th+t), (30-t)//(tw+t)
        style = str(rng.choice(("cap", "comb", "zig") if not test and index == 0 else STYLES))
        n = 3 if dh else int(rng.integers(3, 5))
        gap = int(rng.choice([0, 0, 1]))
        skip = 0 if dh or not test else int(rng.integers(0, 3))
        span = 1 if dh else int(rng.integers(1, 3))
        if n*sw+(n-1)*gap >= nc:
            return None
        required = ((n+skip+span-1)*(sw+gap)+sw if mandatory_growth else
                    (n+skip)*(sw+gap)+1)
        max_start = nc-required
        if max_start < 0:
            return None
        nc = int(rng.integers(required, nc+1))
        max_start = nc-required
        x0 = int(rng.integers(max_start+1))
        dy = int(rng.choice([-1, 0, 0, 0, 1]))
        heights = [sh+i*dh for i in range(n+skip+span)]
        if min(heights) < 1 or max(heights) > th:
            return None
        low = -min(i*dy for i in range(len(heights)))
        upper_extent = max(i*dy+v for i, v in enumerate(heights))
        minimum_rows = upper_extent+low
        if minimum_rows > nr:
            return None
        nr = int(rng.integers(minimum_rows, nr+1))
        high = nr-upper_extent
        if high < low:
            return None
        y0 = int(rng.integers(low, high+1))
        roles = list(map(int, rng.choice(10, 6 if rng.random() < .8 else 7, replace=False)))
        divider, bg, marker = roles[0], roles[1], roles[-1]
        k = 2 if len(roles) == 6 else 3
        ordinary_colors = roles[1:1+k]
        labels = np.concatenate((np.arange(k), rng.integers(k, size=th*tw-k)))
        rng.shuffle(labels)
        ordinary = np.array(ordinary_colors, dtype=np.int8)[labels.reshape(th, tw)]
        inks = roles[1+k:-1]
        colors = tuple(inks[:(2 if len(inks) > 1 and rng.random() < .8 else 1)])
        if dh == 0 and len(colors) == 1 and gap == 0:
            return None
        anchor, flip_r, flip_c, reverse = [bool(rng.integers(2)) for _ in range(4)]
    geo = th, tw, t, nr, nc, divider
    observed, truth = _blank(geo, ordinary), _blank(geo, ordinary)
    target_records = []
    for i in range(n+skip+span):
        hh, yy, xx = sh+i*dh, y0+i*dy, x0+i*(sw+gap)
        if not 1 <= hh <= th:
            return None
        mask = _draw_shape(style, hh, sw)
        mask = mask[::-1 if flip_r else 1, ::-1 if flip_c else 1]
        ar, ac = (th-hh if anchor else 0), (tw-sw if flip_c else 0)
        tile = np.full((th, tw), bg, dtype=np.int8)
        tile[ar:ar+hh, ac:ac+sw][mask] = colors[i % len(colors)]
        for r, c in np.argwhere(mask):
            rr, cc = yy+int(r), xx+int(c)
            if i < n and not (0 <= rr < nr and 0 <= cc < nc):
                return None
            if i < n:
                _stamp(observed, geo, rr, cc, tile)
            _stamp(truth, geo, rr, cc, tile)
        if i >= n+skip:
            target_records.append((yy, min(nr-1, yy+hh-1), xx, min(nc-1, xx+sw-1)))
    r0, r1 = min(v[0] for v in target_records), max(v[1] for v in target_records)
    c0, c1 = target_records[0][2], target_records[-1][3]
    if c0 >= nc or c1 < c0:
        return None
    # Test windows omit at least one macro row. Training's first two windows
    # expose the whole target, including both tile-level and miniature changes.
    if test:
        if r1 <= r0:
            return None
        if rng.integers(2):
            r0 += 1
        else:
            r1 -= 1
    bb = (t+r0*(th+t), t+r1*(th+t)+th-1,
          t+c0*(tw+t), t+c1*(tw+t)+tw-1)
    # Additional fine-scale cuts stay inside tile interiors, preserving lattice.
    if test and th >= 3:
        bb = (bb[0]+1, bb[1], bb[2], bb[3])
    output = _crop(truth, bb)
    if reverse:
        observed, truth = observed[:, ::-1].copy(), truth[:, ::-1].copy()
        output = output[:, ::-1].copy()
        bb = bb[0], bb[1], observed.shape[1]-1-bb[3], observed.shape[1]-1-bb[2]
    if axis:
        observed, output = observed.T.copy(), output.T.copy()
        bb = bb[2], bb[3], bb[0], bb[1]
    for r, c in _corners(bb):
        observed[r, c] = marker
    pair = {"input": observed, "output": output}
    try:
        prediction, info = _analyze(observed)
        assert np.array_equal(prediction, output)
        if not test and index == 0:
            assert not np.array_equal(output, _ablation(info, freeze_geometry=True))
            assert not np.array_equal(output, _ablation(info, freeze_micro=True))
            assert not np.array_equal(output, _ablation(info, solid_macro=True))
            assert not np.array_equal(output, _ablation(info, ink_only=True))
            for transform in (np.fliplr, np.flipud, lambda a: np.rot90(a, 2)):
                assert not np.array_equal(output, _ablation(info, micro_transform=transform))
    except AssertionError:
        return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _checked_int(num_train, 2, 10, "num_train")
    num_test = _checked_int(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    profile, base_axis = int(rng.integers(3)), int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        test = split == "test"
        for index in range(count):
            axis = (base_axis ^ int(test)) if profile == 0 else base_axis if profile == 1 else int(rng.integers(2))
            pair = None
            for _ in range(96):
                candidate = _construct(rng, test=test, index=index, axis=axis)
                if candidate is not None and _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            if pair is None:
                # 254 ordinary patterns provide a finite, distinct pool, larger
                # than the maximum 13 preceding inputs. Each is reconstructible.
                for code in range(1, 255):
                    candidate = _construct(rng, test=test, index=index, axis=axis, fallback=code)
                    if candidate is not None and _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            assert pair is not None, "finite construction pool exhausted"
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for a in pair.values():
                a = np.asarray(a)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((a >= 0) & (a <= 9))
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            predicted, info = _analyze(pair["input"])
            assert np.array_equal(predicted, pair["output"]), "wrong continuation crop"
            assert info["marker"] not in predicted
            m = info["model"]
            assert m["n"] >= 3 and info["target_tiles"]
            ordinary_crop = _crop(_blank(m["geo"], m["ordinary"]), m["bb"])
            if info["transpose"]:
                ordinary_crop = ordinary_crop.T
            assert not np.array_equal(ordinary_crop, predicted), "marker removal is insufficient"
            infos.append(info)
    first = infos[0]
    out = np.asarray(puzzle["train"][0]["output"])
    assert first["model"]["height"][1] != 0
    assert not np.array_equal(out, _ablation(first, freeze_geometry=True)), "A omission witness missing"
    assert not np.array_equal(out, _ablation(first, freeze_micro=True)), "micro-scale B witness missing"
    assert not np.array_equal(out, _ablation(first, solid_macro=True)), "macro-scale B witness missing"
    assert not np.array_equal(out, _ablation(first, ink_only=True)), "whole-tile copying witness missing"
    for transform in (np.fliplr, np.flipud, lambda a: np.rot90(a, 2)):
        assert not np.array_equal(out, _ablation(first, micro_transform=transform)), "orientation witness missing"


def self_test():
    """42 seeds, bounded in-memory diagnostics; no saved payloads or traces."""
    shapes, tiles, styles, directions, deltas, gaps, cycles, keys, prototypes = (set() for _ in range(9))
    totals = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                predicted, info = _analyze(pair["input"])
                m = info["model"]
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
                shapes.add((pair["input"].shape, predicted.shape))
                tiles.add(m["geo"][:3])
                styles.add(info["style"])
                directions.add((info["transpose"], m["right"]))
                deltas.add(m["height"][1])
                gaps.add(m["x"][1]-m["sw"])
                cycles.add(len(m["cycle"]))
                prototypes.add(_key(m["fine_masks"][0]))
                totals["drift"] += m["y"][1] != 0
                totals["multi_target"] += len(info["target_indices"]) > 1
                totals["skipped"] += min(info["target_indices"]) >= m["n"]+1 if m["right"] else max(info["target_indices"]) <= -2
                totals["nonzero_ordinary"] += 0 not in m["ordinary"]
                assert info["marker"] not in predicted
        keys.add(digest.digest())
    assert len(keys) == 42 and len(shapes) >= 35
    assert len(tiles) >= 6 and len(styles) >= 4 and len(directions) == 4
    assert deltas == {-1, 0, 1} and gaps >= {0, 1} and cycles == {1, 2}
    assert len(prototypes) >= 20 and totals["drift"] >= 2 and totals["nonzero_ordinary"] >= 30
    assert totals["multi_target"] >= 2 and totals["skipped"] >= 2
    # Cell-predicate and stroke paths agree over the grammar, including
    # dimensions not necessarily sampled in those 42 puzzles.
    for style in STYLES:
        for h in range(1, 5):
            for w in range(1, 5):
                assert np.array_equal(_draw_shape(style, h, w), _shape(style, h, w))
    for code in (1, 2, 17, 63, 127, 254):
        for test in (False, True):
            assert _construct(np.random.default_rng(13), test=test, index=0, axis=0, fallback=code) is not None
    a, b = generate_puzzle(123, 10, 4), generate_puzzle(123, 10, 4)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, ne in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(902, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    return {"seeds": 42, "pairs": 210, "shape_pairs": len(shapes),
            "tile_geometries": len(tiles), "styles": sorted(styles),
            "directions": len(directions), "height_increments": sorted(deltas),
            "gaps": sorted(gaps), "ink_periods": sorted(cycles),
            "first_miniatures": len(prototypes), "counts": dict(totals)}
