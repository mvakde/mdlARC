"""Fault impression roll: registered evidence, pre-fold census, seam halos.

Only NumPy and the standard library are used. Labels are built from construction
masks; solve recovers everything from input pixels without accessing those masks.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(((0 <= a) & (a <= 9)).all()), "grid colors must be 0..9")
    return a.astype(np.int64, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _bbox(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty stencil")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (diagonal or abs(dy) + abs(dx) == 1)]
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], [p]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                q = (y + dy, x + dx)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.append(q)
        groups.append(group)
    return groups


def _dilate(mask, diagonal=True):
    h, w = mask.shape
    p = np.pad(mask, 1)
    neighbors = [p[y:y+h, x:x+w] for y in range(3) for x in range(3)
                 if diagonal or abs(y-1) + abs(x-1) <= 1]
    return np.logical_or.reduce(neighbors)


def _decode(value):
    """Unique input-only parse and registration, with no labels or palette state."""
    inp = _grid(value)
    _require(len(np.unique(inp)) == 3 and 0 in inp, "expected field, ink and dividers")
    candidates = []
    for vertical in (False, True):
        g = inp.T if vertical else inp
        h, w = g.shape
        for divider in (int(c) for c in np.unique(g) if c != 0):
            cuts = np.flatnonzero(np.all(g == divider, axis=0)).tolist()
            if not 2 <= len(cuts) <= 4:
                continue
            if np.count_nonzero(g == divider) != len(cuts)*h:
                continue
            m = len(cuts) + 1
            if (w - len(cuts)) % m:
                continue
            a = (w - len(cuts)) // m
            if a < 1 or cuts != [(i+1)*a+i for i in range(m-1)]:
                continue
            sheets = np.stack([g[:, i*(a+1):i*(a+1)+a] for i in range(m)])
            colors = set(map(int, np.unique(sheets)))
            if len(colors) != 2 or 0 not in colors or divider in colors:
                continue
            ink = next(iter(colors - {0}))
            observed = sheets == ink
            votes = observed.sum(axis=0)
            if (2*votes == m).any():
                continue
            clean = 2*votes > m
            if not clean.any() or clean.all():
                continue
            errors = observed != clean[None, :, :]
            if (errors.sum(axis=0) > 1).any():
                continue
            holes = clean[None, :, :] & ~observed
            extras = ~clean[None, :, :] & observed
            n = sum(len(_components(x)) for x in holes)
            if n < 1:
                continue
            y, v, x, u = _bbox(clean)
            tile = clean[y:v, x:u].copy()
            folded = holes.any(axis=0)[y:v, x:u].copy()
            if n*(tile.shape[1]+1)-1 > 30:
                continue
            candidates.append({"vertical": vertical, "divider": divider,
                               "ink": ink, "sheets": sheets, "copies": m,
                               "clean": clean, "holes": holes, "extras": extras,
                               "count": n, "bbox": (y, v, x, u),
                               "tile": tile, "folded": folded})
    _require(len(candidates) == 1, "input lacks a unique registered-sheet parse")
    return candidates[0]


def _predict(d):
    """Modular assembly and global dilation, separate from label construction."""
    tile, folded = d["tile"], d["folded"]
    h, a = tile.shape
    length = d["count"]*(a+1)-1
    columns = np.arange(length) % (a+1)
    occupied = columns < a
    indexes = np.minimum(columns, a-1)
    base = tile[:, indexes] & occupied[None, :]
    missing = folded[:, indexes] & occupied[None, :]
    out = np.where(base, d["ink"], 0).astype(np.int64)
    out[missing] = 0
    out[_dilate(missing) & ~missing] = d["divider"]
    return out.T.copy() if d["vertical"] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive output from this input alone; never consult generation or examples."""
    return _predict(_decode(input_grid))


def _stencil(rng, h, a, kind):
    b = np.zeros((h, a), dtype=bool)
    if kind == "rectangle":
        b[:] = True
    elif kind == "notch":
        b[:] = True
        nh = int(rng.integers(1, max(2, h-2)))
        nw = int(rng.integers(1, max(2, a-1)))
        b[:nh, a-nw:] = False
    elif kind == "L":
        thick = int(rng.integers(1, max(2, a)))
        b[:, :thick] = True
        b[-int(rng.integers(2, h)):] = True
    elif kind == "T":
        thick = int(rng.integers(1, max(2, a)))
        start = int(rng.integers(a-thick+1))
        b[:int(rng.integers(1, h))] = True
        b[:, start:start+thick] = True
    elif kind == "cross":
        thick = int(rng.integers(1, max(2, a)))
        start = int(rng.integers(a-thick+1))
        b[:, start:start+thick] = True
        hh = int(rng.integers(1, h))
        top = int(rng.integers(h-hh+1))
        b[top:top+hh] = True
    elif kind == "ring":
        rim = int(rng.integers(1, min(3, (a+1)//2)))
        b[:] = True
        if h > 2*rim and a > 2*rim:
            b[rim:-rim, rim:-rim] = False
    elif kind == "bands":
        y = 0
        while y < h:
            height = int(rng.integers(1, min(5, h-y)+1))
            b[y:y+height] = True
            y += height + int(rng.integers(1, 4))
    elif kind == "stairs":
        y = 0
        widths = sorted(map(int, rng.integers(2, a+1, size=int(rng.integers(2, 5)))))
        widths[-1] = a
        lengths = np.ones(len(widths), dtype=int)
        for _ in range(h-len(widths)):
            lengths[int(rng.integers(len(lengths)))] += 1
        for width, height in zip(widths, lengths):
            b[y:y+height, :width] = True
            y += int(height)
    elif kind == "arms":
        thick = int(rng.integers(1, max(2, a)))
        start = int(rng.integers(a-thick+1))
        b[:, start:start+thick] = True
        for _ in range(int(rng.integers(2, 5))):
            hh = int(rng.integers(1, min(4, h)+1))
            top = int(rng.integers(h-hh+1))
            left = int(rng.integers(start+1))
            right = int(rng.integers(start+thick, a+1))
            b[top:top+hh, left:right] = True
    else:
        raise ValueError("unknown stencil type")
    if rng.random() < .5:
        b = np.flipud(b)
    if rng.random() < .5:
        b = np.fliplr(b)
    y, v, x, u = _bbox(b)
    return b[y:v, x:u].copy()


def _anchor_holes(rng, clean, m, witness, bbox):
    holes = np.zeros((m, *clean.shape), dtype=bool)
    y0, y1, x0, x1 = bbox
    inside = np.zeros_like(clean)
    inside[y0:y1, x0:x1] = True
    first, second = map(int, rng.choice(m, 2, replace=False))
    if witness == 0:
        options = []
        for y in range(y0, y1-1):
            for x in range(x0, x1-1):
                if not clean[y:y+2, x:x+2].all() or (x != x0 and x+2 != x1):
                    continue
                f = np.zeros_like(clean)
                f[y:y+2, x:x+2] = True
                far = inside & ~clean & ~_dilate(f)
                if far.any():
                    options.append((y, x))
        if not options:
            return None
        y, x = options[int(rng.integers(len(options)))]
        if rng.random() < .5:
            holes[first, y:y+2, x] = True
            holes[second, y:y+2, x+1] = True
        else:
            holes[first, y, x:x+2] = True
            holes[second, y+1, x:x+2] = True
    else:
        # Two edge groups with diagonal contact in one sheet, plus a third
        # group on another. All four points form one group after registration.
        offsets = np.array([(0, 0), (0, 1), (1, 2), (1, 1)])
        if rng.random() < .5:
            offsets = offsets[:, ::-1]
        offsets *= rng.choice((-1, 1), size=2)
        offsets -= offsets.min(axis=0)
        hh, ww = offsets.max(axis=0)+1
        options = []
        for y in range(clean.shape[0]-hh+1):
            for x in range(clean.shape[1]-ww+1):
                pts = offsets + [y, x]
                if clean[pts[:, 0], pts[:, 1]].all():
                    options.append((y, x))
        if not options:
            return None
        pts = offsets + options[int(rng.integers(len(options)))]
        holes[first, pts[:3, 0], pts[:3, 1]] = True
        holes[second, pts[3, 0], pts[3, 1]] = True
    return holes


def _random_holes(rng, clean, m, n):
    holes = np.zeros((m, *clean.shape), dtype=bool)
    used = np.zeros_like(clean)
    for _ in range(n):
        placed = False
        for _ in range(70):
            sheet = int(rng.integers(m))
            forbidden = _dilate(holes[sheet], diagonal=False)
            available = clean & ~used & ~forbidden
            candidates = np.argwhere(available)
            if not len(candidates):
                continue
            p = tuple(candidates[int(rng.integers(len(candidates)))])
            group = {p}
            maximum = min(6, max(1, int(clean.sum())//(n+1)))
            target = int(rng.integers(1, maximum+1))
            for _ in range(target-1):
                frontier = set()
                for y, x in group:
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        q = (y+dy, x+dx)
                        if (0 <= q[0] < clean.shape[0] and 0 <= q[1] < clean.shape[1]
                                and available[q] and q not in group):
                            frontier.add(q)
                if not frontier:
                    break
                choices = sorted(frontier)
                group.add(choices[int(rng.integers(len(choices)))])
            for p in group:
                holes[sheet][p] = True
                used[p] = True
            placed = True
            break
        if not placed:
            return None
    if sum(len(_components(x)) for x in holes) != n:
        return None
    return holes


def _extras(rng, clean, holes, witness, bbox):
    m = len(holes)
    extras = np.zeros_like(holes)
    used = holes.any(axis=0)
    y, v, x, u = bbox
    inside = np.zeros_like(clean)
    inside[y:v, x:u] = True
    if witness == 0:
        special = [inside & ~clean & ~_dilate(used), ~inside]
        for i, available in enumerate(special):
            points = np.argwhere(available & ~used)
            if not len(points):
                return None
            p = tuple(points[int(rng.integers(len(points)))])
            # Put the exterior speck on the first source sheet: a literal crop
            # of that sheet visibly disagrees with recovery and tight cropping.
            sheet = 0 if i == 1 else int(rng.integers(m))
            extras[sheet][p] = True
            used[p] = True
    wanted = int(rng.integers(2 if witness is not None else 0, 8))
    for _ in range(max(0, wanted-int(extras.sum()))):
        points = np.argwhere(~clean & ~used)
        if not len(points):
            break
        p = tuple(points[int(rng.integers(len(points)))])
        extras[int(rng.integers(m))][p] = True
        used[p] = True
    return extras


def _construct(clean, holes, extras, ink, divider, vertical):
    """Explicit-mask label construction; does not call decode, predict or solve."""
    m, h, w = holes.shape
    pieces = []
    for i in range(m):
        sheet = np.where(clean, ink, 0).astype(np.int64)
        sheet[holes[i]] = 0
        sheet[extras[i]] = ink
        pieces.append(sheet)
        if i < m-1:
            pieces.append(np.full((h, 1), divider, dtype=np.int64))
    inp = np.concatenate(pieces, axis=1)
    y, v, x, u = _bbox(clean)
    tile = np.where(clean[y:v, x:u], ink, 0).astype(np.int64)
    folded = holes.any(axis=0)[y:v, x:u]
    tile[folded] = 0
    n = sum(len(_components(g)) for g in holes)
    tile_pieces, hole_pieces = [], []
    for i in range(n):
        tile_pieces.append(tile.copy())
        hole_pieces.append(folded.copy())
        if i < n-1:
            tile_pieces.append(np.zeros((v-y, 1), dtype=np.int64))
            hole_pieces.append(np.zeros((v-y, 1), dtype=bool))
    out = np.concatenate(tile_pieces, axis=1)
    missing = np.concatenate(hole_pieces, axis=1)
    for r, c in np.argwhere(missing):
        out[max(0, r-1):r+2, max(0, c-1):c+2] = divider
    out[missing] = 0
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _make_pair(rng, n, kinds, palette, vertical, witness=None):
    ink, divider = palette
    for _ in range(100):
        m = int(rng.integers(3, 6))
        panel_max = 31//m-1
        axial_max = min(panel_max, 31//n-1, 6 if witness is not None else 9)
        if axial_max < 3:
            continue
        a = int(rng.integers(3, axial_max+1))
        h = int(rng.integers(max(6, a+2), 13)) if witness is not None else int(rng.integers(4, 25))
        kind = "notch" if witness == 0 else "stairs" if witness == 1 else str(rng.choice(kinds))
        tile = _stencil(rng, h, a, kind)
        h, a = tile.shape
        # Cross-axis field margin guarantees the ink cannot imitate a divider.
        ypad = int(rng.integers(1, min(6, 30-h)+1))
        xpad = int(rng.integers(min(4, panel_max-a)+1))
        top = int(rng.integers(ypad+1))
        left = int(rng.integers(xpad+1))
        clean = np.zeros((h+ypad, a+xpad), dtype=bool)
        clean[top:top+h, left:left+a] = tile
        bbox = _bbox(clean)
        holes = (_anchor_holes(rng, clean, m, witness, bbox) if witness is not None
                 else _random_holes(rng, clean, m, n))
        if holes is None:
            continue
        if sum(len(_components(g)) for g in holes) != n:
            continue
        extras = _extras(rng, clean, holes, witness, bbox)
        if extras is None:
            continue
        pair = _construct(clean, holes, extras, ink, divider, vertical)
        d = _decode(pair["input"])
        if np.array_equal(d["clean"], clean) and np.array_equal(d["holes"], holes):
            return pair
    raise RuntimeError("bounded construction could not place the required evidence")


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10),
                               ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palettes = [tuple(map(int, rng.choice(np.arange(1, 10), 2, replace=False)))
                for _ in range(int(rng.integers(2, 5)))]
    # Both visible nonzero roles change across the two learning witnesses.
    options = [(s, d) for s in range(1, 10) for d in range(1, 10)
               if s != d and s != palettes[0][0] and d != palettes[0][1]]
    palettes[1] = options[int(rng.integers(len(options)))]
    regime = int(rng.integers(3))
    vertical = bool(rng.integers(2))
    all_kinds = ("rectangle", "notch", "L", "T", "cross", "ring", "bands", "stairs", "arms")
    train_kinds = ("rectangle", "notch", "L", "T") if regime == 1 else all_kinds
    test_kinds = ("ring", "bands", "arms", "cross") if regime == 1 else all_kinds
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for i in range(amount):
            witness = i if split == "train" and i < 2 else None
            n = 2+i if witness is not None else int(rng.integers(1, 4)) if split == "train" else (
                int(rng.integers(4, 8)) if regime == 0 else int(rng.integers(1, 7)))
            if regime == 2:
                pair_vertical = vertical if split == "train" else not vertical
            else:
                pair_vertical = bool(rng.integers(2))
            palette = palettes[i] if witness is not None else palettes[int(rng.integers(len(palettes)))]
            for _ in range(30):
                pair = _make_pair(rng, n, train_kinds if split == "train" else test_kinds,
                                  palette, pair_vertical, witness)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    n = d["count"]
    eight = sum(len(_components(g, True)) for g in d["holes"])
    folded_n = len(_components(d["folded"]))
    y, v, x, u = d["bbox"]
    inside_extras = d["extras"].any(axis=0)[y:v, x:u]
    far_extra = bool((inside_extras & ~_dilate(d["folded"])).any())
    fold = d["folded"]
    return {"groups": n, "eight_groups": eight, "folded_groups": folded_n,
            "hole_area": int(d["holes"].sum()), "extra_area": int(d["extras"].sum()),
            "copies": d["copies"], "tile_shape": d["tile"].shape,
            "far_inside_extra": far_extra,
            "axial_edge_hole": bool(fold[:, 0].any() or fold[:, -1].any()),
            "palette": (d["ink"], d["divider"]), "vertical": d["vertical"],
            "stencil_components": len(_components(d["tile"])),
            "sizes": tuple(sorted(len(g) for sheet in d["holes"] for g in _components(sheet))),
            "margin": (y, d["clean"].shape[0]-v, x, d["clean"].shape[1]-u)}


def _validate(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, records = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            canonical = out.T if d["vertical"] else out
            tile, folded = d["tile"], d["folded"]
            h, a = tile.shape
            n = d["count"]
            _require(canonical.shape == (h, n*(a+1)-1), "wrong roll extent")
            # Audit returned cells directly, without calling solve or the label
            # painter. Build spatial masks and explicitly inspect each cell.
            base, missing = np.zeros(canonical.shape, bool), np.zeros(canonical.shape, bool)
            for k in range(n):
                start = k*(a+1)
                base[:, start:start+a] = tile
                missing[:, start:start+a] = folded
            for r in range(h):
                for c in range(canonical.shape[1]):
                    neighborhood = missing[max(0, r-1):r+2, max(0, c-1):c+2]
                    expected = (0 if missing[r, c] else d["divider"] if neighborhood.any()
                                else d["ink"] if base[r, c] else 0)
                    _require(int(canonical[r, c]) == expected, "output violates census/folding/halo rule")
            f = _features(d)
            halo = _dilate(missing) & ~missing
            cross_halo = _dilate(missing, diagonal=False) & ~missing
            gap_columns = np.arange(canonical.shape[1]) % (a+1) == a
            f["diagonal_halo_witness"] = bool((halo & ~cross_halo).any())
            f["seam_halo_witness"] = bool(halo[:, gap_columns].any())
            # Suppressing A's negative-error treatment while retaining B's
            # actual count restores the stencil completely and changes cells.
            restored_only = np.where(base, d["ink"], 0)
            f["omit_A_changes"] = not np.array_equal(restored_only, canonical)
            # Suppressing quantity-driven repetition changes the roll extent.
            f["omit_B_changes"] = n > 1
            if split == "train":
                training.append((d, f))
            records.append((split, d, f))
    fs = [f for d, f in training]
    _require(any(a["hole_area"] == b["hole_area"] and a["groups"] != b["groups"]
                 for a in fs for b in fs), "training confounds component count and area")
    _require(any(f["groups"] > f["folded_groups"] for f in fs), "no pre-fold census witness")
    _require(any(f["groups"] != f["eight_groups"] for f in fs), "no edge-connectivity witness")
    _require(any(f["groups"] != f["copies"] for f in fs), "sheet count shortcut survives")
    _require(any(f["far_inside_extra"] for f in fs), "no visible asymmetric cleanup witness")
    _require(any(f["axial_edge_hole"] and f["groups"] > 1 for f in fs), "no seam halo witness")
    _require(any(f["seam_halo_witness"] for f in fs), "isolated-tile halo shortcut survives")
    _require(any(f["diagonal_halo_witness"] for f in fs), "four-neighbor halo shortcut survives")
    _require(len({f["palette"] for f in fs}) >= 2, "no visible palette reassignment")
    _require(any(not d["tile"].all() and any(f["margin"]) for d, f in training),
             "no field-versus-fault and cropping witness")
    _require(any(d["extras"][0].any() and
                 (d["extras"][0].sum() > d["extras"][0][d["bbox"][0]:d["bbox"][1],
                                                         d["bbox"][2]:d["bbox"][3]].sum())
                 for d, f in training), "no literal observed crop counterexample")
    for symmetry in range(1, 8):
        _require(any(not np.array_equal(d["tile"], np.rot90(
            np.fliplr(d["tile"]) if symmetry >= 4 else d["tile"], symmetry % 4))
                     for d, f in training), "stencil orientation is not established")
    # Concrete omissions of the inherited mechanisms must change training.
    _require(any(f["omit_B_changes"] for f in fs), "omitting B's repetition has no effect")
    _require(all(f["omit_A_changes"] for f in fs), "omitting A's holes has no effect")
    return records


def validate_generated_puzzle(puzzle) -> None:
    """Check visible family invariants, labels, and collective training evidence."""
    try:
        _validate(puzzle)
    except (ValueError, TypeError, IndexError, KeyError) as error:
        raise AssertionError(str(error)) from error


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed variation, metamorphic checks, and count coverage."""
    shapes, silhouettes, palettes, counts, copies, axes, sizes = (set() for _ in range(7))
    signatures, stencil_components = set(), set()
    rows = []
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_key(p))
        for split, d, f in _validate(p):
            rows.append((split, f))
            silhouettes.add(_key(d["tile"]))
            shapes.add(d["clean"].shape)
            palettes.add(f["palette"])
            counts.add(f["groups"])
            copies.add(f["copies"])
            axes.add(f["vertical"])
            sizes.update(f["sizes"])
            stencil_components.add(f["stencil_components"])
        for pair in p["train"]+p["test"]:
            inp, out = pair["input"], pair["output"]
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(solve(inp.T), out.T)
            assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
            # Reassign nonzero roles; zero's family role is fixed.
            mapping = np.r_[0, np.roll(np.arange(1, 10), 1)]
            assert np.array_equal(solve(mapping[inp]), mapping[out])
        if seed in (0, 7, 41):
            assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(silhouettes) >= 90
    assert len(palettes) >= 30
    assert copies == {3, 4, 5} and axes == {False, True}
    assert {1, 2, 3, 4, 5, 6, 7} <= counts
    assert min(sizes) == 1 and max(sizes) >= 5
    assert max(stencil_components) >= 2
    assert any(s == "test" and f["groups"] == 1 for s, f in rows)
    # Supported stencils extend beyond the thick rectilinear sampling vocabulary.
    # This narrow, disconnected fixture has two positive errors, three missing
    # groups, a genuine internal blank and no geometric reconstruction shortcut.
    clean = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0],
                      [0, 1, 0, 1, 1, 0],
                      [0, 1, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0]], dtype=bool)
    holes = np.zeros((3, *clean.shape), bool)
    holes[0, 1, 1:3] = True
    holes[1, 2, 4] = holes[2, 4, 3] = True
    extras = np.zeros_like(holes)
    extras[0, 5, 5] = extras[1, 3, 3] = True
    fixture = _construct(clean, holes, extras, 4, 7, False)
    assert _decode(fixture["input"])["count"] == 3
    assert np.array_equal(solve(fixture["input"]), fixture["output"])
    # An axial-length-one crop also follows exactly the same rule.
    thin = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0],
                     [0, 1, 0], [0, 0, 0]], dtype=bool)
    hh, ee = np.zeros((3, *thin.shape), bool), np.zeros((3, *thin.shape), bool)
    hh[0, 1, 1] = hh[1, 3, 1] = True
    ee[2, 0, 0] = True
    fixture = _construct(thin, hh, ee, 8, 1, True)
    assert np.array_equal(solve(fixture["input"]), fixture["output"])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7301, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupt["test"][0]["output"].flat[0] = (int(corrupt["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed label")
    return {"seeds": 42, "pairs": len(rows), "distinct_puzzles": len(signatures),
            "sheet_shapes": len(shapes), "stencil_structures": len(silhouettes),
            "palettes": len(palettes), "group_counts": sorted(counts),
            "sheet_counts": sorted(copies), "component_sizes": sorted(sizes)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
