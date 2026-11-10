"""Count-keyed motif transplantation followed by a bottom-anchored row shear.

Only the standard library and NumPy are used. No parent module, saved example,
seed, or sampling policy is consulted by solve(). self_test() writes no files.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 744049303, 991453246, 113564509, 876074281, 180206995,
              423168817, 768044315, 508640271, 9562995, 785025066, 576419400,
              691976027, 439129539, 987830652, 567141006, 973564428, 140728551,
              869819315, 735890156, 542400880, 459082587, 574461677, 390622483,
              800825379, 752189845, 744221072, 937139944, 788983014, 74326157,
              53254831, 982105183, 522121680, 902232903, 574172905, 754956835,
              639884054, 610846400, 162886927, 284407998, 439000675, 14253673)


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _FOUR
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _variants(mask):
    # Rectangular motifs also preserve orientation; quarter turns exchange sides.
    return [np.rot90(mask, k) for k in range(4)] + [np.rot90(mask.T, k) for k in range(4)]


def _asymmetric(mask):
    return all(x.shape != mask.shape or not np.array_equal(x, mask)
               for x in _variants(mask)[1:])


def _safe_seeds(h, w):
    """A bounded constructive independent set, retaining connected foreground."""
    best = np.zeros((h, w), dtype=bool)
    for parity in (0, 1):
        cells = [(r, c) for r in range(h) for c in range(w) if (r + c) % 2 == parity]
        for order in (cells, cells[::-1]):
            mask = np.zeros((h, w), dtype=bool)
            for p in order:
                mask[p] = True
                if len(_components(np.pad(~mask, 1, constant_values=True))) != 1:
                    mask[p] = False
            if mask.sum() > best.sum():
                best = mask
    return best


def _capacity(th, tw):
    return min(8, int(_safe_seeds(th - 4, tw - 4).sum()))


def _isolated(rng, h, w, count):
    if count == 0:
        return np.zeros((h, w), dtype=bool)
    for _ in range(6):
        mask = np.zeros((h, w), dtype=bool)
        for i in rng.permutation(h * w):
            r, c = divmod(int(i), w)
            if any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                   for dr, dc in _FOUR):
                continue
            mask[r, c] = True
            if len(_components(np.pad(~mask, 1, constant_values=True))) != 1:
                mask[r, c] = False
                continue
            if int(mask.sum()) == count:
                return mask
    available = np.argwhere(_safe_seeds(h, w))
    assert len(available) >= count
    mask = np.zeros((h, w), dtype=bool)
    for i in rng.permutation(len(available))[:count]:
        mask[tuple(available[int(i)])] = True
    if rng.integers(2):
        mask = np.flipud(mask)
    if rng.integers(2):
        mask = np.fliplr(mask)
    return mask.copy()


def _rectangles(rng, h, w, count):
    boxes = [(0, h, 0, w)]
    while len(boxes) < count:
        cuts = []
        for i, (a, b, c, d) in enumerate(boxes):
            cuts.extend((i, 0, p) for p in range(a + 1, b - 1))
            cuts.extend((i, 1, p) for p in range(c + 1, d - 1))
        if not cuts:
            return None
        i, axis, p = _pick(rng, cuts)
        a, b, c, d = boxes.pop(i)
        boxes.extend([(a, p, c, d), (p + 1, b, c, d)] if axis == 0
                     else [(a, b, c, p), (a, b, p + 1, d)])
    mask = np.zeros((h, w), dtype=bool)
    for a, b, c, d in boxes:
        if rng.random() < 0.45:
            a = int(rng.integers(a, b))
            b = int(rng.integers(a + 1, b + 1))
            c = int(rng.integers(c, d))
            d = int(rng.integers(c + 1, d + 1))
        mask[a:b, c:d] = True
    return mask


def _cavity_mask(rng, mh, mw, count):
    if count == 0:
        return np.zeros((mh, mw), dtype=bool)
    h, w = mh - 2, mw - 2
    if count == 1 and rng.random() < 0.35:
        return np.pad(np.ones((h, w), dtype=bool), 1)
    if rng.random() < 0.45:
        inner = _rectangles(rng, h, w, count)
        if inner is not None:
            if rng.integers(2):
                inner = np.flipud(inner)
            if rng.integers(2):
                inner = np.fliplr(inner)
            return np.pad(inner, 1)
    mask = np.pad(_isolated(rng, h, w, count), 1)
    target = int(rng.integers(count, h * w + 1))
    for _ in range(2):
        for i in rng.permutation(h * w):
            r, c = divmod(int(i), w)
            r, c = r + 1, c + 1
            if int(mask.sum()) >= target:
                return mask
            if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _FOUR):
                continue
            mask[r, c] = True
            if len(_components(mask)) != count or len(_components(~mask)) != 1:
                mask[r, c] = False
    return mask


def _two_cavity_witness(rng, mh, mw):
    # The singleton touches the domino diagonally: two four-cavities, one eight-cavity.
    mask = np.zeros((mh, mw), dtype=bool)
    mask[1, 1] = mask[2, 2] = mask[2, 3] = True
    for i in rng.permutation((mh - 2) * (mw - 2)):
        if rng.random() < 0.6:
            continue
        r, c = divmod(int(i), mw - 2)
        r, c = r + 1, c + 1
        trial = mask.copy()
        trial[r, c] = True
        if (trial.sum() < (mh - 2) * (mw - 2) and len(_components(trial)) == 2
                and len(_components(trial, True)) == 1 and len(_components(~trial)) == 1
                and _asymmetric(trial)):
            mask = trial
    variants = [x for x in _variants(mask) if x.shape == mask.shape]
    return _pick(rng, variants).copy()


def _free_origins(boxes, h, w):
    rr, cc = np.indices((31 - h, 31 - w))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, bh, bw in boxes:
        free &= (rr + h < r) | (rr > r + bh) | (cc + w < c) | (cc > c + bw)
    return np.argwhere(free)


def _fallback_pack(config):
    th, tw, rows, cols = config
    boxes = [(0, 0, rows * th, cols * tw)]
    for _ in range(rows * cols):
        free = _free_origins(boxes, th - 2, tw - 2)
        if not len(free):
            return None
        r, c = map(int, free[0])
        boxes.append((r, c, th - 2, tw - 2))
    return boxes


_CONFIGS = tuple((th, tw, r, c) for th in range(6, 11) for tw in range(6, 11)
                 for r, c in ((1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 2))
                 if r * th + c * tw - 1 <= 30 and r * c <= _capacity(th, tw) + 1
                 and _fallback_pack((th, tw, r, c)) is not None)


def _place_scene(rng, mosaic, motifs, background, tile_order, witness):
    h, w = mosaic.shape
    mh, mw = next(iter(motifs.values())).shape
    boxes = None
    for _ in range(16):
        r = int(rng.integers(31 - h))
        c = int(rng.integers(31 - w))
        trial = [(r, c, h, w)]
        for _ in motifs:
            free = _free_origins(trial, mh, mw)
            if not len(free):
                break
            a, b = map(int, _pick(rng, free))
            trial.append((a, b, mh, mw))
        if len(trial) == len(motifs) + 1:
            boxes = trial
            break
    if boxes is None:
        boxes = _fallback_pack((mh + 2, mw + 2, h // (mh + 2), w // (mw + 2)))
    assert boxes is not None
    # Motif reading order is intentionally different from tile reading order in the witness.
    motif_boxes = sorted(boxes[1:])
    order = tile_order[1:] + tile_order[:1] if witness else list(map(int, rng.permutation(tile_order)))
    canvas = np.full((30, 30), background, dtype=np.int8)
    r, c = boxes[0][:2]
    canvas[r:r + h, c:c + w] = mosaic
    for (r, c, bh, bw), k in zip(motif_boxes, order):
        canvas[r:r + bh, c:c + bw] = motifs[k]
    low_r, low_c = min(b[0] for b in boxes), min(b[1] for b in boxes)
    high_r = max(b[0] + b[2] for b in boxes)
    high_c = max(b[1] + b[3] for b in boxes)
    if rng.random() < 0.10:
        return canvas
    # Tight crops, uneven margins, interior mosaics, and edge-touching objects all occur.
    low_r = int(rng.integers(low_r + 1)) if rng.random() < 0.5 else low_r
    low_c = int(rng.integers(low_c + 1)) if rng.random() < 0.5 else low_c
    high_r = int(rng.integers(high_r, 31)) if rng.random() < 0.5 else high_r
    high_c = int(rng.integers(high_c, 31)) if rng.random() < 0.5 else high_c
    return canvas[low_r:high_r, low_c:high_c].copy()


def _shear(completed, background):
    h, w = completed.shape
    result = np.full((h, h + w - 1), background, dtype=np.int8)
    for r in range(h):
        result[r, h - 1 - r:h - 1 - r + w] = completed[r]
    return result


def _make_pair(rng, config, counts, background, witness):
    th, tw, rows, cols = config
    mh, mw, n = th - 2, tw - 2, rows * cols
    order = list(map(int, rng.permutation(counts)))
    pool = list(map(int, rng.permutation([c for c in range(10) if c != background])))
    pool = pool[:int(rng.integers(2, 10))]
    carriers = dict(zip(order, map(int, rng.choice(pool, size=n, replace=True))))
    if witness:
        carriers[1], carriers[2] = pool[:2]
    motifs, masks = {}, {}
    for k in order:
        mask = _cavity_mask(rng, mh, mw, k)
        if witness and k == 1:
            mask = np.pad(np.ones((mh - 2, mw - 2), dtype=bool), 1)
        if witness and k == 2:
            mask = _two_cavity_witness(rng, mh, mw)
        color = _pick(rng, [c for c in pool if c != carriers[k]])
        if witness and k in (1, 2):
            color = carriers[3 - k]
        motif = np.full((mh, mw), color, dtype=np.int8)
        motif[mask] = background
        masks[k], motifs[k] = mask, motif
    mosaic = np.empty((rows * th, cols * tw), dtype=np.int8)
    completed = np.empty_like(mosaic)
    for i, k in enumerate(order):
        r, c = (i // cols) * th, (i % cols) * tw
        tile = np.full((th, tw), carriers[k], dtype=np.int8)
        if int(masks[k].sum()) == k and rng.random() < 0.16:
            clues = masks[k].copy()
        else:
            clues = _isolated(rng, mh, mw, k)
        tile[1:-1, 1:-1][clues] = background
        mosaic[r:r + th, c:c + tw] = tile
        tile[:] = carriers[k]
        tile[1:-1, 1:-1] = np.where(masks[k], carriers[k], motifs[k])
        completed[r:r + th, c:c + tw] = tile
    inp = _place_scene(rng, mosaic, motifs, background, order, witness)
    return {"input": inp, "output": _shear(completed, background)}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError("grid must be an integer rectangle of size 1..30 with colors 0..9")
    return a.astype(np.int8, copy=False)


def _rim(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _decode_with_background(grid, bg):
    objects = sorted(_components(grid != bg), key=len, reverse=True)
    assert 3 <= len(objects) <= 5
    boxes = [_box(x) for x in objects]
    a, b, c, d = boxes[0]
    mosaic = grid[a:b, c:d]
    dimensions = {(v - u, y - x) for u, v, x, y in boxes[1:]}
    assert len(dimensions) == 1
    mh, mw = next(iter(dimensions))
    th, tw = mh + 2, mw + 2
    assert 6 <= th <= 10 and 6 <= tw <= 10
    assert mosaic.shape[0] % th == 0 and mosaic.shape[1] % tw == 0
    rows, cols = mosaic.shape[0] // th, mosaic.shape[1] // tw
    n = rows * cols
    assert n == len(objects) - 1 and 2 <= n <= 4
    assert sum(mosaic.shape) - 1 <= 30
    for i, (u, v, x, y) in enumerate(boxes):
        for e, f, g, h in boxes[i + 1:]:
            assert v < e or f < u or y < g or h < x
    motifs, motif_boxes = {}, {}
    for u, v, x, y in boxes[1:]:
        motif = grid[u:v, x:y]
        colors = np.unique(motif[motif != bg])
        assert len(colors) == 1 and np.all(_rim(motif) == colors[0])
        k = len(_components(motif == bg))
        assert k not in motifs
        motifs[k], motif_boxes[k] = motif.copy(), (u, v, x, y)
    carriers, counts, clues = [], [], []
    for r in range(rows):
        for col in range(cols):
            tile = mosaic[r * th:(r + 1) * th, col * tw:(col + 1) * tw]
            carrier = int(tile[0, 0])
            assert carrier != bg and np.all(_rim(tile) == carrier)
            assert set(map(int, np.unique(tile))) <= {bg, carrier}
            clue = tile[1:-1, 1:-1] == bg
            assert not np.any(clue[:-1] & clue[1:]) and not np.any(clue[:, :-1] & clue[:, 1:])
            k = int(clue.sum())
            assert k in motifs and int(motifs[k][0, 0]) != carrier
            counts.append(k)
            carriers.append(carrier)
            clues.append(clue.copy())
    assert len(set(counts)) == n and set(counts) == set(motifs)
    return {"background": bg, "tile_shape": (th, tw), "layout": (rows, cols),
            "mosaic_box": boxes[0], "motifs": motifs, "motif_boxes": motif_boxes,
            "carriers": carriers, "tile_counts": counts, "clues": clues}


def _decode(grid):
    candidates = []
    for bg in map(int, np.unique(grid)):
        try:
            candidates.append(_decode_with_background(grid, bg))
        except (AssertionError, ValueError, IndexError):
            continue
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous background/mosaic interpretation")
    return candidates[0]


def _complete(model, assignment=None):
    th, tw = model["tile_shape"]
    rows, cols = model["layout"]
    bg = model["background"]
    motifs = model["motifs"] if assignment is None else assignment
    result = np.empty((th * rows, tw * cols), dtype=np.int8)
    for i, k in enumerate(model["tile_counts"]):
        carrier = model["carriers"][i]
        tile = np.full((th, tw), carrier, dtype=np.int8)
        tile[1:-1, 1:-1] = np.where(motifs[k] == bg, carrier, motifs[k])
        r, c = (i // cols) * th, (i % cols) * tw
        result[r:r + th, c:c + tw] = tile
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the given grid, complete its mosaic, and shear complete rows."""
    model = _decode(_grid(input_grid))
    return _shear(_complete(model), model["background"])


def _features(model):
    bg = model["background"]
    masks = {k: m == bg for k, m in model["motifs"].items()}
    areas = {k: int(m.sum()) for k, m in masks.items()}
    return {
        "area_inversion": any(a < b and areas[a] > areas[b] for a in masks for b in masks),
        "connectivity": any(len(_components(m, True)) < k for k, m in masks.items()),
        "asymmetry": any(_asymmetric(m) for m in masks.values()),
        "nonpositional": any(not np.array_equal(clue, masks[k])
                              for clue, k in zip(model["clues"], model["tile_counts"]) if k),
        "order_mismatch": [k for k, _ in sorted(model["motif_boxes"].items(), key=lambda x: x[1])]
                          != model["tile_counts"],
        "color_mismatch": any(model["carriers"][i] in
                              [int(m[0, 0]) for m in model["motifs"].values()]
                              for i in range(len(model["carriers"]))),
    }


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            model = _decode(inp)
            completed = _complete(model)
            h, w = completed.shape
            bg = model["background"]
            assert out.shape == (h, h + w - 1), "wrong widened crop dimensions"
            for r in range(h):
                start = h - 1 - r
                assert np.array_equal(out[r, start:start + w], completed[r]), "row vector changed"
                assert np.all(out[r, :start] == bg) and np.all(out[r, start + w:] == bg)
            assert np.array_equal(out[-1, :w], completed[-1]), "bottom anchor moved"
            assert np.count_nonzero(out != bg) == completed.size, "content lost or stray padding"
            # Each parent must be causally necessary in every example.
            a, b, c, d = model["mosaic_box"]
            assert not np.array_equal(out, _shear(inp[a:b, c:d], bg)), "insertion is inactive"
            ks = sorted(model["motifs"])
            wrong = dict(model["motifs"])
            wrong[ks[0]], wrong[ks[1]] = wrong[ks[1]], wrong[ks[0]]
            assert not np.array_equal(out, _shear(_complete(model, wrong), bg)), "matching is inactive"
            assert out.shape != completed.shape, "shear is inactive"
            analyses[split].append(model)
    features = [_features(m) for m in analyses["train"]]
    for feature in ("area_inversion", "connectivity", "asymmetry", "nonpositional",
                    "order_mismatch", "color_mismatch"):
        assert any(f[feature] for f in features), "training lacks " + feature
    assert len({m["tile_shape"] for m in analyses["train"]}) >= 2
    assert any(0 in m["motifs"] for m in analyses["train"]), "zero case not demonstrated"
    assert any(m["layout"][0] > 1 for m in analyses["train"]), "global row origin not demonstrated"
    return analyses


def validate_generated_puzzle(puzzle):
    """Check labels, scene grammar, parent necessity, and demonstration witnesses."""
    try:
        _validate(puzzle)
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise AssertionError("invalid generated puzzle: " + str(error)) from error


def _integer(name, value, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(name + " has an invalid integer value")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt, ne = _integer("num_train", num_train, 2, 10), _integer("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    curriculum = _pick(rng, ("layout", "growth", "aspect", "shrink"))
    fixed_bg = int(rng.integers(10))
    vary_bg = bool(rng.random() < 0.8)
    puzzle, seen, train_configs = {"train": [], "test": []}, set(), []
    previous_bg = fixed_bg
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            witness = split == "train" and index == 0
            eligible = list(_CONFIGS)
            if curriculum == "layout":
                eligible = [x for x in eligible if (x[3] == 1 if split == "train"
                                                   else x[3] > 1)]
            elif curriculum == "growth":
                eligible = [x for x in eligible if (x[2] * x[3] == 2 if split == "train"
                                                   else x[2] * x[3] >= 3)]
            elif curriculum == "aspect":
                eligible = [x for x in eligible if (x[0] == x[1] if split == "train"
                                                   else x[0] != x[1])]
            else:
                eligible = [x for x in eligible if (x[2] * x[3] >= 3 if split == "train"
                                                   else x[2] * x[3] == 2)]
            if witness:
                eligible = [x for x in eligible if min(x[:2]) >= 7]
            if split == "train" and index == 1:
                eligible = [x for x in eligible if x[:2] != train_configs[0][:2]]
                if train_configs[0][2] == 1:
                    eligible = [x for x in eligible if x[2] > 1]
            config = _pick(rng, eligible)
            n, cap = config[2] * config[3], _capacity(*config[:2])
            required = [1, 2] if witness else ([0] if split == "train" and index == 1 else [])
            if witness and n > 2 and rng.random() < 0.5:
                required.append(0)
            allowed = [k for k in range(cap + 1) if k not in required]
            counts = required + list(map(int, rng.choice(allowed, size=n - len(required), replace=False)))
            bg = _pick(rng, [c for c in range(10) if c != previous_bg]) if vary_bg else fixed_bg
            previous_bg = bg
            for _ in range(12):
                pair = _make_pair(rng, config, counts, bg, witness)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                train_configs.append(config)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, sequential, in-memory 42-seed audit; returns compact metrics."""
    shapes, layouts, counts, patterns, backgrounds, palettes, signatures = (set() for _ in range(7))
    squares = rectangles = multirow = area_witnesses = connectivity_witnesses = 0
    max_area = repeated_carriers = nonconsecutive = width30 = 0
    shifts = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        analyses = _validate(puzzle)
        signatures.add(_fingerprint(puzzle))
        tr, te = analyses["train"], analyses["test"]
        if min(len(m["motifs"]) for m in te) > max(len(m["motifs"]) for m in tr):
            shifts.add("growth")
        if max(len(m["motifs"]) for m in te) < min(len(m["motifs"]) for m in tr):
            shifts.add("shrink")
        if all(m["tile_shape"][0] == m["tile_shape"][1] for m in tr) and any(
                m["tile_shape"][0] != m["tile_shape"][1] for m in te):
            shifts.add("aspect")
        if any(m["layout"] not in {x["layout"] for x in tr} for m in te):
            shifts.add("layout")
        area_witnesses += any(_features(m)["area_inversion"] for m in tr)
        connectivity_witnesses += any(_features(m)["connectivity"] for m in tr)
        for split in ("train", "test"):
            for pair, m in zip(puzzle[split], analyses[split]):
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                bg = m["background"]
                backgrounds.add(bg)
                shapes.add(tuple(pair["input"].shape))
                layouts.add(m["tile_shape"] + m["layout"])
                ks = tuple(sorted(m["motifs"]))
                counts.add(ks)
                nonconsecutive += max(ks) - min(ks) + 1 > len(ks)
                repeated_carriers += len(set(m["carriers"])) < len(m["carriers"])
                squares += m["tile_shape"][0] == m["tile_shape"][1]
                rectangles += m["tile_shape"][0] != m["tile_shape"][1]
                multirow += m["layout"][0] > 1
                width30 += pair["output"].shape[1] == 30
                palettes.add(len(np.unique(pair["input"])))
                for motif in m["motifs"].values():
                    mask = motif == bg
                    patterns.add((mask.shape, min(x.tobytes() for x in _variants(mask))))
                    max_area = max(max_area, max(map(len, _components(mask)), default=0))
    assert len(signatures) == 42 and len(backgrounds) >= 8
    assert len(shapes) >= 25 and len(layouts) >= 25 and len(patterns) >= 80
    assert len(counts) >= 20 and len(palettes) >= 5
    assert rectangles >= 30 and squares >= 20 and multirow >= 30
    assert area_witnesses == connectivity_witnesses == 42 and max_area >= 16
    assert nonconsecutive >= 10 and repeated_carriers >= 10 and width30 >= 1
    assert shifts == {"growth", "shrink", "aspect", "layout"}
    assert _fingerprint(generate_puzzle(17)) == _fingerprint(generate_puzzle(17))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(2523, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Global palette permutation must commute with the entire transformation.
    sample = generate_puzzle(0, 2, 1)
    lookup = np.array([7, 4, 9, 0, 6, 8, 1, 5, 3, 2], dtype=np.int8)
    recolored = {s: [{k: lookup[v] for k, v in p.items()} for p in sample[s]] for s in sample}
    validate_generated_puzzle(recolored)
    for pair in recolored["test"]:
        assert np.array_equal(solve(pair["input"]), pair["output"])
    # Input-only solving does not mutate its input. Removing outer neutral margins
    # and translating all objects together cannot affect the cropped output.
    for pair in sample["train"] + sample["test"]:
        before = pair["input"].copy()
        predicted = solve(before)
        assert np.array_equal(before, pair["input"])
        assert np.array_equal(predicted, pair["output"])
        bg = _decode(before)["background"]
        rr, cc = np.where(before != bg)
        tight = before[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        assert np.array_equal(solve(tight), pair["output"])
        spare_h, spare_w = 30 - tight.shape[0], 30 - tight.shape[1]
        moved = np.pad(tight, ((min(1, spare_h), 0), (min(2, spare_w), 0)),
                       constant_values=bg)
        assert np.array_equal(solve(moved), pair["output"])
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad padding accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "tile_shape_layouts": len(layouts),
            "count_sets": len(counts), "cavity_shapes_modulo_D4": len(patterns),
            "backgrounds": len(backgrounds), "palette_sizes": sorted(palettes),
            "square_pairs": squares, "rectangular_pairs": rectangles,
            "multirow_pairs": multirow, "maximum_cavity_area": max_area,
            "repeated_carrier_pairs": repeated_carriers, "nonconsecutive_count_pairs": nonconsecutive,
            "width_30_pairs": width30, "training_area_witnesses": area_witnesses,
            "training_connectivity_witnesses": connectivity_witnesses, "shifts": sorted(shifts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
