"""Missing motifs in a mirrored quilt; NumPy and the standard library only.

Generation retains one puzzle at a time and writes no files. The input-only
solver uses frame geometry, mirror votes and a fine-scale missing-material halo.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(binary):
    remaining = set(map(tuple, np.argwhere(binary)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _frame_box(cells):
    points = np.asarray(cells)
    r, c = map(int, points.min(axis=0))
    v, u = map(int, points.max(axis=0) + 1)
    if min(v-r, u-c) < 4:
        return None
    expected = {(y, x) for y in range(r, v) for x in range(c, u)
                if y in (r, v-1) or x in (c, u-1)}
    return (r, v, c, u) if set(cells) == expected else None


def _recover(observed):
    if any(n not in (4, 6, 8, 10) for n in observed.shape):
        raise ValueError("mask dimensions must be even, 4..10")
    a = observed.astype(np.int8)
    votes = a + a[::-1] + a[:, ::-1] + a[::-1, ::-1]
    if np.any(votes == 2):
        raise ValueError("tied reflection orbit")
    clean = votes >= 3
    if not clean.any() or clean.all():
        raise ValueError("both intended mask states must be present")
    return clean


def _read_input(input_grid):
    a = _grid(input_grid)
    b = int(a[0, 0])
    if not (np.all(a[0] == b) and np.all(a[-1] == b)
            and np.all(a[:, 0] == b) and np.all(a[:, -1] == b)):
        raise ValueError("a uniform exterior field border is required")
    candidates = []
    for h in map(int, np.unique(a)):
        if h == b:
            continue
        comps = _components(a == h)
        if len(comps) != 2:
            continue
        boxes = [_frame_box(cells) for cells in comps]
        if None in boxes:
            continue
        p, q = boxes
        if not (p[1] < q[0] or q[1] < p[0]
                or p[3] < q[2] or q[3] < p[2]):
            continue
        inside = np.zeros_like(a, dtype=bool)
        for r, v, c, u in boxes:
            inside[r:v, c:u] = True
        if np.any(a[~inside] != b):
            continue
        interiors = [a[r+1:v-1, c+1:u-1].copy() for r, v, c, u in boxes]
        for mi, ti in ((0, 1), (1, 0)):
            mask_grid, tile = interiors[mi], interiors[ti]
            signal = set(map(int, np.unique(mask_grid))) - {b}
            payload = set(map(int, np.unique(tile))) - {b}
            if (len(signal) != 1 or len(payload) < 2 or h in signal | payload
                    or signal & payload or not all(2 <= n <= 5 for n in tile.shape)):
                continue
            s = next(iter(signal))
            observed = mask_grid == s
            try:
                clean = _recover(observed)
            except ValueError:
                continue
            if any(m*t > 30 for m, t in zip(observed.shape, tile.shape)):
                continue
            candidates.append({"field": b, "halo": h, "signal": s,
                               "observed": observed, "clean": clean, "tile": tile,
                               "missing": clean & ~observed,
                               "extras": ~clean & observed,
                               "mask_box": boxes[mi], "tile_box": boxes[ti]})
    if len(candidates) != 1:
        raise ValueError("input must have one framed binary/multicolor reading")
    return candidates[0]


def _dilate(binary, diagonal=True):
    p = np.pad(binary, 1)
    result = np.zeros_like(binary)
    h, w = binary.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if diagonal or abs(dr) + abs(dc) <= 1:
                result |= p[1+dr:1+dr+h, 1+dc:1+dc+w]
    return result


def _fine_sets(info, solid_missing=False):
    tile, b = info["tile"], info["field"]
    sh, sw = tile.shape
    mh, mw = info["observed"].shape
    r, c = np.indices((mh*sh, mw*sw))
    material = tile[r % sh, c % sw] != b
    missing = info["missing"][r // sh, c // sw]
    d = missing if solid_missing else missing & material
    valid = (info["clean"] & info["observed"])[r // sh, c // sw]
    quilt = np.where(valid, tile[r % sh, c % sw], b).astype(np.int8)
    return quilt, d


def _predict(info, diagonal=True, solid_missing=False):
    quilt, d = _fine_sets(info, solid_missing)
    quilt[_dilate(d, diagonal) & ~d] = info["halo"]
    return quilt


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive everything from this input, including the annotation color."""
    return _predict(_read_input(input_grid))


def _render_constructed(clean, missing, extras, tile, b, h, s):
    """Label construction uses explicit masks and loops, without solve()."""
    observed = clean.copy()
    observed[missing], observed[extras] = False, True
    sh, sw = tile.shape
    out = np.full((clean.shape[0]*sh, clean.shape[1]*sw), b, dtype=np.int8)
    d = np.zeros_like(out, dtype=bool)
    for r, c in np.argwhere(clean):
        ys, xs = slice(r*sh, (r+1)*sh), slice(c*sw, (c+1)*sw)
        if missing[r, c]:
            d[ys, xs] = tile != b
        else:
            out[ys, xs] = tile
    for r, c in np.argwhere(d):
        out[max(0, r-1):r+2, max(0, c-1):c+2] = h
    out[d] = b
    return np.where(observed, s, b).astype(np.int8), out


def _orbit(p, shape):
    r, c = p
    return min(r, shape[0]-1-r), min(c, shape[1]-1-c)


def _mirrors(master):
    top = np.concatenate((master, master[:, ::-1]), axis=1)
    return np.concatenate((top, top[::-1]), axis=0)


def _master(rng, shape, family):
    h, w = shape
    r, c = np.indices(shape)
    a = np.zeros(shape, dtype=bool)
    if family == "rectangle":
        y, x = int(rng.integers(h)), int(rng.integers(w))
        v, u = int(rng.integers(y+1, h+1)), int(rng.integers(x+1, w+1))
        a[y:v, x:u] = True
    elif family == "bands":
        axis = int(rng.integers(2))
        choices = rng.random(shape[axis]) < .5
        a = np.broadcast_to(choices[:, None] if axis == 0 else choices[None, :], shape).copy()
    elif family == "stairs":
        lengths = np.sort(rng.integers(0, w+1, size=h))
        a = c < lengths[:, None]
    elif family == "arms":
        y, x = int(rng.integers(h)), int(rng.integers(w))
        a[y, :] = True
        a[:, x] = True
        if rng.random() < .5:
            a[:y+1, :x+1] = True
    elif family == "cavity":
        a[:] = True
        y, x = int(rng.integers(h)), int(rng.integers(w))
        v, u = int(rng.integers(y+1, h+1)), int(rng.integers(x+1, w+1))
        a[y:v, x:u] = False
    elif family == "islands":
        a = rng.random(shape) < float(rng.uniform(.25, .8))
    else:
        raise ValueError("unknown master geometry")
    if rng.random() < .5:
        a = a[::-1]
    if rng.random() < .5:
        a = a[:, ::-1]
    return a.copy()


def _symmetries(a):
    return [f(np.rot90(a, k)) for k in range(4) for f in (lambda x: x, np.fliplr)]


def _asymmetric(a):
    return all(not np.array_equal(a, t) for t in _symmetries(a)[1:])


def _tile(rng, shape, b, payload, witness):
    h, w = shape
    r, c = np.indices(shape)
    family = str(rng.choice(("bands", "diagonal", "rings", "repeat", "random")))
    n = len(payload)
    for _ in range(64):
        if witness:
            indices = rng.integers(n, size=shape)
        elif family == "bands":
            indices = (r if rng.random() < .5 else c) + int(rng.integers(n))
        elif family == "diagonal":
            indices = r + c*int(rng.integers(1, n+1)) + int(rng.integers(n))
        elif family == "rings":
            indices = np.minimum.reduce((r, c, h-1-r, w-1-c)) + int(rng.integers(n))
            if min(shape) == 2:
                # A two-cell-wide rectangle has only one concentric layer.
                indices = rng.integers(n, size=shape)
        elif family == "repeat":
            little = rng.integers(n, size=(2, 2))
            indices = little[r % 2, c % 2]
        else:
            indices = rng.integers(n, size=shape)
        tile = np.asarray(payload, dtype=np.int8)[indices % n].copy()
        holes = np.zeros(shape, dtype=bool)
        if witness == "single":
            holes[1, 1] = True
            holes[0, 1] = True
        elif witness == "cluster":
            holes[0, 1] = True if w == 3 else False
            if w != 3:
                holes[1, 0] = True
        elif rng.random() < .8:
            holes = rng.random(shape) < float(rng.uniform(.05, .4))
        # Visible corner support also makes the diagonal halo witness explicit.
        holes[0, 0] = holes[0, -1] = holes[-1, 0] = holes[-1, -1] = False
        tile[holes] = b
        if len(set(map(int, np.unique(tile))) - {b}) < 2:
            continue
        if witness and not _asymmetric(tile):
            continue
        return tile
    raise RuntimeError("failed to sample a multicolor motif")


def _choose_from_orbits(rng, available, count, used, preferred=None):
    result = np.zeros_like(available)
    cells = np.argwhere(available)
    order = rng.permutation(len(cells))
    for i in order:
        p = tuple(map(int, cells[int(i)]))
        orbit = _orbit(p, available.shape)
        if orbit in used or (preferred is not None and not preferred[p]):
            continue
        result[p] = True
        used.add(orbit)
        if int(result.sum()) == count:
            break
    return result if int(result.sum()) == count else None


def _damage(rng, clean, witness, is_test):
    h, w = clean.shape
    missing = np.zeros_like(clean)
    used = set()
    if witness == "single":
        if not (clean[0, 0] and bool(clean[0, 1]) != bool(clean[1, 0])):
            return None
        corner = ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))[int(rng.integers(4))]
        missing[corner] = True
        used.add(_orbit(corner, clean.shape))
    elif witness == "cluster":
        choices = []
        for r in range(h-1):
            for c in range(w-1):
                block = [(r, c), (r+1, c), (r, c+1), (r+1, c+1)]
                for omitted in range(4):
                    pts = [p for i, p in enumerate(block) if i != omitted]
                    if (all(clean[p] for p in pts)
                            and len({_orbit(p, clean.shape) for p in pts}) == 3):
                        choices.append(pts)
        if not choices:
            return None
        points = choices[int(rng.integers(len(choices)))]
        for p in points:
            missing[p] = True
            used.add(_orbit(p, clean.shape))
    else:
        available = int(clean[:h//2, :w//2].sum())
        count = int(rng.integers(1 if is_test else 0, min(6 if is_test else 4, available)+1))
        if count:
            missing = _choose_from_orbits(rng, clean, count, used)
            if missing is None:
                return None
    available_extra = int((~clean[:h//2, :w//2]).sum())
    count = int(rng.integers(1 if witness or not missing.any() else 0,
                             min(4, available_extra)+1))
    extras = np.zeros_like(clean)
    if count:
        # Teaching witnesses always expose at least one erasure outside halos.
        far = ~_dilate(missing)
        if witness:
            first = _choose_from_orbits(rng, ~clean, 1, used, far)
            if first is None:
                return None
            extras |= first
            count -= 1
        if count:
            rest = _choose_from_orbits(rng, ~clean, count, used)
            if rest is None:
                return None
            extras |= rest
    return missing, extras


def _frame(interior, color):
    out = np.full((interior.shape[0]+2, interior.shape[1]+2), color, dtype=np.int8)
    out[1:-1, 1:-1] = interior
    return out


def _place_objects(rng, board, tile, b, h):
    a, t = _frame(board, h), _frame(tile, h)
    if rng.random() < .5:
        a, t = t, a
    horizontal = bool(rng.integers(2))
    gap = int(rng.integers(1, 5))
    if horizontal:
        minimum_h, minimum_w = max(a.shape[0], t.shape[0])+2, a.shape[1]+t.shape[1]+gap+2
    else:
        minimum_h, minimum_w = a.shape[0]+t.shape[0]+gap+2, max(a.shape[1], t.shape[1])+2
    hh = minimum_h + int(rng.integers(min(6, 30-minimum_h)+1))
    ww = minimum_w + int(rng.integers(min(6, 30-minimum_w)+1))
    out = np.full((hh, ww), b, dtype=np.int8)
    if horizontal:
        x = int(rng.integers(1, ww-a.shape[1]-t.shape[1]-gap))
        pos = [(int(rng.integers(1, hh-a.shape[0])), x),
               (int(rng.integers(1, hh-t.shape[0])), x+a.shape[1]+gap)]
    else:
        y = int(rng.integers(1, hh-a.shape[0]-t.shape[0]-gap))
        pos = [(y, int(rng.integers(1, ww-a.shape[1]))),
               (y+a.shape[0]+gap, int(rng.integers(1, ww-t.shape[1])))]
    for obj, (r, c) in zip((a, t), pos):
        out[r:r+obj.shape[0], c:c+obj.shape[1]] = obj
    return out


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _shapes(rng, regime, witness, is_test):
    if witness == "single":
        return (6, 6), (3, 3)
    if witness == "cluster":
        return ((4, 6), (2, 3)) if rng.random() < .5 else ((6, 4), (3, 2))
    if is_test and regime == "mask_growth":
        m = tuple(map(int, rng.choice((8, 10), size=2)))
        s = tuple(map(int, rng.choice((2, 3), size=2)))
    elif is_test and regime == "tile_growth":
        m = tuple(map(int, rng.choice((4, 6), size=2)))
        s = tuple(map(int, rng.choice((4, 5), size=2)))
    elif regime == "aspect":
        m = (int(rng.choice((8, 10))), 4) if is_test else (4, 6)
        feasible = [(h, w) for h in range(2, 6) for w in range(2, 6)
                    if m[0]*h <= 30 and m[1]*w <= 30]
        s = feasible[int(rng.integers(len(feasible)))]
    else:
        m = tuple(map(int, rng.choice((4, 6), size=2)))
        s = tuple(map(int, rng.choice((2, 3), size=2)))
    return m, s


def _make_pair(rng, regime, palette, varied, witness, is_test):
    families = ("rectangle", "bands", "stairs", "arms", "cavity", "islands")
    for _ in range(96):
        mshape, tshape = _shapes(rng, regime, witness, is_test)
        master = _master(rng, (mshape[0]//2, mshape[1]//2), str(rng.choice(families)))
        if not 2 <= int(master.sum()) < master.size:
            continue
        clean = _mirrors(master)
        damage = _damage(rng, clean, witness, is_test)
        if damage is None:
            continue
        missing, extras = damage
        colors = rng.permutation(10) if varied else palette
        b, h, s = map(int, colors[:3])
        payload = colors[3:3+int(rng.integers(2, 5))]
        tile = _tile(rng, tshape, b, payload, witness)
        board, out = _render_constructed(clean, missing, extras, tile, b, h, s)
        inp = _place_objects(rng, board, tile, b, h)
        # An input reconstruction gate is separate from the explicit painter.
        info = _read_input(inp)
        if not np.array_equal(info["clean"], clean):
            raise AssertionError("mirror recovery disagrees with construction")
        if not np.array_equal(_predict(info), out):
            raise AssertionError("fine-scale prediction disagrees with construction")
        return {"input": inp, "output": out}
    raise RuntimeError("bounded attempts exhausted while constructing teaching evidence")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(("mask_growth", "tile_growth", "aspect")))
    palette = rng.permutation(10)
    varied = bool(rng.integers(2))
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            witness = ("single", "cluster")[i] if split == "train" and i < 2 else ""
            for _ in range(32):
                pair = _make_pair(rng, regime, palette, varied, witness, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input attempts exhausted")
    rng.shuffle(result["train"])
    validate_generated_puzzle(result)
    return result


def _expand(mask, tile, b):
    """Direct unchanged block substitution, also used for shortcut challenges."""
    sh, sw = tile.shape
    r, c = np.indices((mask.shape[0]*sh, mask.shape[1]*sw))
    return np.where(mask[r//sh, c//sw], tile[r % sh, c % sw], b).astype(np.int8)


def _alternatives(info):
    b, h, tile = info["field"], info["halo"], info["tile"]
    coarse_halo = _dilate(info["missing"]) & ~info["missing"]
    coarse_first = _expand(info["clean"] & ~info["missing"], tile, b)
    coarse_first[np.repeat(np.repeat(coarse_halo, tile.shape[0], axis=0), tile.shape[1], axis=1)] = h
    with_extras = _predict(info)
    extra_pixels = np.repeat(np.repeat(info["extras"], tile.shape[0], axis=0), tile.shape[1], axis=1)
    raw = _expand(info["observed"], tile, b)
    with_extras[extra_pixels] = raw[extra_pixels]
    reflected = dict(info, tile=np.fliplr(tile))
    return {"omit_A": raw, "fill_omissions": _expand(info["clean"], tile, b),
            "coarse_halo": coarse_first, "solid_missing": _predict(info, solid_missing=True),
            "four_neighbors": _predict(info, diagonal=False), "retain_extras": with_extras,
            "reflect_tiles": _predict(reflected)}


def _features(info):
    missing, extras, tile = info["missing"], info["extras"], info["tile"]
    _, d = _fine_sets(info)
    fine_halo = _dilate(d) & ~d
    native_in_missing = (np.repeat(np.repeat(missing, tile.shape[0], axis=0), tile.shape[1], axis=1)
                         & ~d & fine_halo)
    edge = np.zeros_like(missing)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    return {"mask_shape": missing.shape, "tile_shape": tile.shape,
            "clean_pattern": _key(info["clean"]),
            "tile_pattern": _normalized(tile, info["field"]),
            "holes": int(missing.sum()), "extras": int(extras.sum()),
            "native_holes": int((tile == info["field"]).sum()),
            "aperture_highlight": bool(native_in_missing.any()),
            "edge": bool((missing & edge).any()),
            "orthogonal": bool((missing[1:] & missing[:-1]).any() or (missing[:, 1:] & missing[:, :-1]).any()),
            "diagonal": bool((missing[1:, 1:] & missing[:-1, :-1]).any()
                             or (missing[1:, :-1] & missing[:-1, 1:]).any()),
            "far_extra": bool((extras & ~_dilate(missing)).any()),
            "asymmetric": _asymmetric(tile), "halo_pixels": int(fine_halo.sum())}


def _normalized(a, b):
    roles = {b: 0}
    return a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.ravel())


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise AssertionError("splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    evidence = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read_input(inp)
            expected = _predict(info)
            assert np.array_equal(expected, out), "incorrect quilt or fine halo"
            assert info["missing"].any() or info["extras"].any(), "no damage"
            _, d = _fine_sets(info)
            assert np.all(out[d] == info["field"]), "an omission was filled"
            assert info["signal"] not in out, "coarse instructions leaked into output"
            features = _features(info)
            records[split].append((info, features))
            if split == "train":
                for name, prediction in _alternatives(info).items():
                    evidence[name] += not np.array_equal(prediction, out)
    assert all(evidence[name] for name in ("omit_A", "fill_omissions", "coarse_halo", "solid_missing",
                                         "four_neighbors", "retain_extras", "reflect_tiles")), evidence
    fs = [f for _, f in records["train"]]
    assert any(f["holes"] == 1 and f["edge"] and f["aperture_highlight"] and f["asymmetric"] for f in fs)
    assert any(f["orthogonal"] and f["diagonal"] for f in fs), "teach merged clusters"
    assert any(f["far_extra"] for f in fs), "teach erasure away from halo"
    assert any(f["tile_shape"][0] != f["tile_shape"][1] for f in fs)
    assert len({f["tile_shape"] for f in fs}) >= 2
    assert len({f["mask_shape"] for f in fs}) >= 2
    assert len({f["tile_pattern"] for f in fs}) >= 2
    assert len({f["clean_pattern"] for f in fs}) >= 2
    # The generated test shift must be visible in the returned dimensions.
    assert all(max(f["mask_shape"]) > 6 or min(f["tile_shape"]) >= 4
               for _, f in records["test"])
    return None


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded in-memory tests; aggregate diversity evidence, never sample dumps."""
    shapes, masks, tiles, palettes, signatures = set(), set(), set(), set(), set()
    holes, extras, flags = set(), set(), Counter()
    counts_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                f = _features(info)
                shapes.add((pair["input"].shape, pair["output"].shape))
                masks.add(f["clean_pattern"])
                tiles.add(f["tile_pattern"])
                palettes.add((info["field"], info["signal"], info["halo"]))
                holes.add(f["holes"])
                extras.add(f["extras"])
                for flag in ("edge", "orthogonal", "diagonal", "far_extra", "aperture_highlight", "asymmetric"):
                    flags[flag] += int(f[flag])
                digest.update(str(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                if seed < 3:
                    for transform in (np.fliplr, np.transpose, lambda a: np.rot90(a)):
                        assert np.array_equal(solve(transform(pair["input"])), transform(pair["output"]))
        signatures.add(digest.digest())
        if seed < 3:
            assert _equal(puzzle, generate_puzzle(seed)), "seed replay failed"
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123456, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
            counts_checked += 1
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.2):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError("illegal seed accepted")
    broken = generate_puzzle(7, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed output accepted")
    tied = np.zeros((4, 4), dtype=bool)
    tied[0, 0] = tied[-1, 0] = True
    try:
        _recover(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous reflection orbit accepted")
    assert len(signatures) == 42
    assert len(shapes) >= 100 and len(masks) >= 50 and len(tiles) >= 100
    assert len(palettes) >= 30 and len(holes) >= 5 and len(extras) >= 4
    assert all(flags.values())
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "shape_combinations": len(shapes), "clean_masks": len(masks),
            "normalized_tiles": len(tiles), "role_palettes": len(palettes),
            "hole_counts": sorted(holes), "extra_counts": sorted(extras),
            "features": dict(flags), "legal_counts_checked": counts_checked,
            "independent_solver": False, "blind_prediction": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
