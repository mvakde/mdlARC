"""Topology-indexed constellations: local graph degree selects a cavity glyph.

Only NumPy and the standard library are used. No parent code or files are read.
Generation paints from construction data; solve recovers all data from pixels.
Nothing is written to disk by this module, including self_test().
"""
from __future__ import annotations

import hashlib
import numbers
import numpy as np


CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = CARDINAL + DIAGONAL if diagonal else CARDINAL
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _symmetries(a):
    return [np.rot90(a, k) for k in range(4)] + [np.rot90(a.T, k) for k in range(4)]


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in _symmetries(a)[1:])


def _degrees(points, diagonal=False):
    points = set(points)
    steps = CARDINAL + DIAGONAL if diagonal else CARDINAL
    return {p: sum((p[0] + dr, p[1] + dc) in points for dr, dc in steps)
            for p in sorted(points)}


def _degree_array(mask):
    p = np.pad(mask, 1)
    return (p[:-2, 1:-1].astype(int) + p[2:, 1:-1]
            + p[1:-1, :-2] + p[1:-1, 2:])


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _separated(a, b):
    # Boxes use exclusive bottom/right, so strict inequality leaves a blank band.
    r, c, h, w = a
    y, x, bh, bw = b
    return r + h < y or y + bh < r or c + w < x or x + bw < c


def _decode(inp):
    inp = _grid(inp)
    values, populations = np.unique(inp, return_counts=True)
    assert len(values) >= 4
    assert np.sum(populations == populations.max()) == 1
    bg = int(values[np.argmax(populations)])
    catalog, singletons, boxes = {}, [], []
    for cells in _components(inp != bg):
        if len(cells) == 1:
            singletons.append(cells[0])
            continue
        r, c, b, e = _bbox(cells)
        glyph = inp[r:b, c:e]
        ink = int(glyph[0, 0])
        assert min(glyph.shape) >= 3 and ink != bg
        assert set(map(int, np.unique(glyph))) <= {bg, ink}
        assert np.all(_perimeter(glyph) == ink)
        assert len(cells) == int(np.count_nonzero(glyph != bg))
        holes = glyph == bg
        k = len(_components(holes))
        assert 0 <= k <= 4 and k not in catalog
        catalog[k] = {"ink": ink, "holes": holes.copy(), "mask": ~holes,
                      "box": (r, c, b-r, e-c)}
        boxes.append((r, c, b-r, e-c))
    assert 2 <= len(catalog) <= 5 and len(singletons) >= 3
    assert len({entry["holes"].shape for entry in catalog.values()}) == 1
    ink_to_degree = {v["ink"]: k for k, v in catalog.items()}
    assert len(ink_to_degree) == len(catalog)
    anchors = [p for p in singletons if int(inp[p]) in ink_to_degree]
    assert len(anchors) == 1
    anchor_pixel = anchors[0]
    marker_values = {int(inp[p]) for p in singletons if p != anchor_pixel}
    assert len(marker_values) == 1
    marker = marker_values.pop()
    assert marker not in ink_to_degree and marker != bg
    pr, pc, pb, pe = _bbox(singletons)
    map_box = (pr, pc, pb-pr, pe-pc)
    all_boxes = boxes + [map_box]
    assert all(_separated(a, b) for i, a in enumerate(all_boxes) for b in all_boxes[i+1:])
    assert all((r-pr) % 2 == 0 and (c-pc) % 2 == 0 for r, c in singletons)
    points = tuple(sorted(((r-pr)//2, (c-pc)//2) for r, c in singletons))
    anchor = ((anchor_pixel[0]-pr)//2, (anchor_pixel[1]-pc)//2)
    degrees = _degrees(points)
    assert set(degrees.values()) == set(catalog)
    anchor_degree = ink_to_degree[int(inp[anchor_pixel])]
    assert degrees[anchor] == anchor_degree
    h, w = next(iter(catalog.values()))["mask"].shape
    mh, mw = (pb-pr)//2+1, (pe-pc)//2+1
    tr, tc = catalog[anchor_degree]["box"][:2]
    top, left = tr-anchor[0]*h, tc-anchor[1]*w
    assert 0 <= top and top+mh*h <= inp.shape[0]
    assert 0 <= left and left+mw*w <= inp.shape[1]
    return {"bg": bg, "marker": marker, "catalog": catalog, "h": h, "w": w,
            "points": points, "degrees": degrees, "anchor": anchor,
            "anchor_degree": anchor_degree, "map_shape": (mh, mw),
            "map_box": map_box, "top": top, "left": left, "shape": inp.shape}


def _reconstruct(meta):
    """Solver uses Boolean Kronecker substitution after input-only decoding."""
    h, w = meta["h"], meta["w"]
    coarse_shape = meta["map_shape"]
    block = np.full((coarse_shape[0]*h, coarse_shape[1]*w), meta["bg"], dtype=np.int8)
    for degree, entry in meta["catalog"].items():
        sites = np.zeros(coarse_shape, dtype=bool)
        for p, d in meta["degrees"].items():
            if d == degree and p != meta["anchor"]:
                sites[p] = True
        block[np.kron(sites, entry["mask"])] = entry["ink"]
    ar, ac = meta["anchor"]
    anchor_block = block[ar*h:(ar+1)*h, ac*w:(ac+1)*w]
    anchor_block[meta["catalog"][meta["anchor_degree"]]["mask"]] = meta["marker"]
    out = np.full(meta["shape"], meta["bg"], dtype=np.int8)
    r, c = meta["top"], meta["left"]
    out[r:r+block.shape[0], c:c+block.shape[1]] = block
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output from one input; no generation state is used."""
    try:
        return _reconstruct(_decode(input_grid))
    except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("input violates the topology-indexed constellation rule") from error


def _alternative(meta, mode):
    """Finite competing rules used only for demonstration-quality checks."""
    out = np.full(meta["shape"], meta["bg"], dtype=np.int8)
    h, w = meta["h"], meta["w"]
    tr, tc = meta["catalog"][meta["anchor_degree"]]["box"][:2]
    anchor = meta["anchor"]
    if mode == "center":
        anchor = tuple((n-1)//2 for n in meta["map_shape"])
    ph, pw = (2, 2) if mode == "pitch_two" else (h, w)
    top, left = tr-anchor[0]*ph, tc-anchor[1]*pw
    degree_map = _degrees(meta["points"], diagonal=True) if mode == "eight_neighbors" else meta["degrees"]
    if mode == "distance":
        degree_map = {meta["anchor"]: 0}
        queue = [meta["anchor"]]
        occupied = set(meta["points"])
        for r, c in queue:
            for dr, dc in CARDINAL:
                q = (r+dr, c+dc)
                if q in occupied and q not in degree_map:
                    degree_map[q] = degree_map[(r, c)]+1
                    queue.append(q)
    order = sorted(meta["catalog"], key=lambda k: (int(meta["catalog"][k]["holes"].sum()), k))
    area_lookup = dict(zip(sorted(meta["catalog"]), order))
    for p in meta["points"]:
        k = degree_map.get(p, -1)
        if mode == "one_glyph":
            k = meta["anchor_degree"]
        elif mode == "area_rank":
            k = area_lookup[k]
        if k not in meta["catalog"]:
            continue
        entry = meta["catalog"][k]
        mask = np.ones((h, w), dtype=bool) if mode == "solid" else entry["mask"]
        ink = meta["marker"] if p == meta["anchor"] and mode != "no_inversion" else entry["ink"]
        # A wrong registration/pitch may leave the canvas; clip only hypotheses.
        r, c = top+p[0]*ph, left+p[1]*pw
        for y, x in np.argwhere(mask):
            if 0 <= r+y < out.shape[0] and 0 <= c+x < out.shape[1]:
                out[r+y, c+x] = ink
    return out


def _training_coverage(metas, pairs):
    assert set().union(*(set(m["catalog"]) for m in metas)) == set(range(5))
    assert len({m["anchor"][0] for m in metas}) >= 2
    assert len({m["anchor"][1] for m in metas}) >= 2
    assert len({m["h"] for m in metas}) >= 2 and len({m["w"] for m in metas}) >= 2
    assert any(_asymmetric(e["holes"]) for m in metas for e in m["catalog"].values())
    assert any(len(_components(e["holes"], True)) < k
               for m in metas for k, e in m["catalog"].items())
    assert any(a < b and m["catalog"][a]["holes"].sum() > m["catalog"][b]["holes"].sum()
               for m in metas for a in m["catalog"] for b in m["catalog"])
    for mode in ("one_glyph", "area_rank", "pitch_two", "center", "solid",
                 "eight_neighbors", "distance", "no_inversion"):
        assert any(not np.array_equal(_alternative(m, mode), np.asarray(p["output"]))
                   for m, p in zip(metas, pairs)), "training permits " + mode


def validate_generated_puzzle(puzzle):
    """Validate pixels, exact outputs, and meaningful training witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_metas = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            meta = _decode(inp)
            assert inp.shape == out.shape and np.array_equal(_reconstruct(meta), out)
            assert not np.array_equal(inp, out)
            expected_ink = sum(int(meta["catalog"][d]["mask"].sum()) for d in meta["degrees"].values())
            assert np.count_nonzero(out != meta["bg"]) == expected_ink
            # Exact per-site mask preservation, including recoloring and holes.
            for (r, c), d in meta["degrees"].items():
                rr, cc = meta["top"]+r*meta["h"], meta["left"]+c*meta["w"]
                block = out[rr:rr+meta["h"], cc:cc+meta["w"]]
                entry = meta["catalog"][d]
                ink = meta["marker"] if (r, c) == meta["anchor"] else entry["ink"]
                assert np.array_equal(block, np.where(entry["mask"], ink, meta["bg"]))
            if split == "train":
                train_metas.append(meta)
    _training_coverage(train_metas, puzzle["train"])


def _cavities(rng, h, w, k, witness=False):
    holes = np.zeros((h, w), dtype=bool)
    if k == 0:
        return holes
    if k == 1 and (witness or rng.random() < 0.35):
        holes[1:-1, 1:-1] = True
        return holes
    if k == 2 and witness:
        holes[1, 1] = holes[2, 2] = holes[2, 3] = True
        options = [v for v in _symmetries(holes) if v.shape == (h, w)]
        return _pick(rng, options).copy()
    # Sample separated seeds, allowing both parities. Keep connected ink.
    for _ in range(24):
        holes[:] = False
        for index in rng.permutation((h-2)*(w-2)):
            r, c = divmod(int(index), w-2)
            r, c = r+1, c+1
            if any(holes[r+dr, c+dc] for dr, dc in CARDINAL):
                continue
            holes[r, c] = True
            if len(_components(~holes)) != 1:
                holes[r, c] = False
            if holes.sum() == k:
                break
        if holes.sum() == k:
            break
    else:
        holes[:] = False
        # Four mutually separated interior corners always fit h,w >= 5.
        corners = [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]
        for index in rng.permutation(4)[:k]:
            holes[corners[int(index)]] = True
    target = int(rng.integers(k, (h-2)*(w-2)+1))
    for _ in range((h-2)*(w-2)):
        if holes.sum() >= target:
            break
        candidates = []
        for r in range(1, h-1):
            for c in range(1, w-1):
                if holes[r, c] or not any(holes[r+dr, c+dc] for dr, dc in CARDINAL):
                    continue
                holes[r, c] = True
                if len(_components(holes)) == k and len(_components(~holes)) == 1:
                    candidates.append((r, c))
                holes[r, c] = False
        if not candidates:
            break
        holes[_pick(rng, candidates)] = True
    assert len(_components(holes)) == k and len(_components(~holes)) == 1
    return holes


def _layout(rng, dims, teaching=False, previous_anchor=None):
    mh, mw = dims
    for _ in range(256):
        mask = rng.random(dims) < rng.uniform(0.22, 0.86)
        if not all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())):
            continue
        degrees = set(_degree_array(mask)[mask].tolist())
        if mask.sum() < 3 or len(degrees) < 2:
            continue
        if teaching and degrees != set(range(5)):
            continue
        if previous_anchor is not None and not any(
                r != previous_anchor[0] and c != previous_anchor[1] for r, c in np.argwhere(mask)):
            continue
        return mask
    if teaching:
        # All five local valencies in a tight 4x4 pattern, only a finite fallback.
        mask = np.array([[1, 1, 1, 0], [1, 1, 1, 0],
                         [0, 1, 0, 0], [0, 0, 0, 1]], dtype=bool)
        assert dims == (4, 4)
        return _pick(rng, _symmetries(mask)).copy()
    # Tight path with endpoint degree one and internal degree two.
    mask = np.zeros(dims, dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    assert mask.sum() >= 3
    options = [mask, np.flipud(mask), np.fliplr(mask), np.flipud(np.fliplr(mask))]
    if previous_anchor is not None:
        options = [a for a in options if any(r != previous_anchor[0] and c != previous_anchor[1]
                                           for r, c in np.argwhere(a))]
    return _pick(rng, options).copy()


def _free_origins(boxes, h, w):
    rr, cc = np.indices((31-h, 31-w))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, bh, bw in boxes:
        free &= (rr+h < r) | (rr > r+bh) | (cc+w < c) | (cc > c+bw)
    return np.argwhere(free)


def _place(rng, layout, anchor, holes, colors):
    h, w = next(iter(holes.values())).shape
    mh, mw = layout.shape
    points = list(map(tuple, np.argwhere(layout).tolist()))
    degrees = _degrees(points)
    ak = degrees[anchor]
    ph, pw = 2*mh-1, 2*mw-1
    for _ in range(64):
        top = int(rng.integers(31-mh*h))
        left = int(rng.integers(31-mw*w))
        tr, tc = top+anchor[0]*h, left+anchor[1]*w
        boxes = [(tr, tc, h, w)]
        free = _free_origins(boxes, ph, pw)
        if not len(free):
            continue
        pr, pc = map(int, _pick(rng, free))
        boxes.append((pr, pc, ph, pw))
        entries = {ak: (tr, tc)}
        for k in rng.permutation([k for k in holes if k != ak]):
            free = _free_origins(boxes, h, w)
            if not len(free):
                break
            r, c = map(int, _pick(rng, free))
            entries[int(k)] = (r, c)
            boxes.append((r, c, h, w))
        if len(entries) == len(holes):
            break
    else:
        raise RuntimeError("bounded catalog packing exhausted")
    bg, marker, inks = colors
    inp = np.full((30, 30), bg, dtype=np.int8)
    out = np.full_like(inp, bg)
    for k, (r, c) in entries.items():
        inp[r:r+h, c:c+w] = np.where(holes[k], bg, inks[k])
    for p, k in degrees.items():
        inp[pr+2*p[0], pc+2*p[1]] = inks[k] if p == anchor else marker
        r, c = top+p[0]*h, left+p[1]*w
        out[r:r+h, c:c+w] = np.where(holes[k], bg, marker if p == anchor else inks[k])
    # Include the complete output extent when optionally cropping blank margins.
    boxes.append((top, left, mh*h, mw*w))
    low_r, low_c = min(b[0] for b in boxes), min(b[1] for b in boxes)
    high_r, high_c = max(b[0]+b[2] for b in boxes), max(b[1]+b[3] for b in boxes)
    r0, c0 = int(rng.integers(low_r+1)), int(rng.integers(low_c+1))
    r1, c1 = int(rng.integers(high_r, 31)), int(rng.integers(high_c, 31))
    inp, out = inp[r0:r1, c0:c1].copy(), out[r0:r1, c0:c1].copy()
    values, pops = np.unique(inp, return_counts=True)
    assert int(values[np.argmax(pops)]) == bg and np.sum(pops == pops.max()) == 1
    return {"input": inp, "output": out}


def _options(regime, split):
    choices = []
    for h in range(5, 8):
        for w in range(5, 8):
            for mh in range(1, min(6, 30//h)+1):
                for mw in range(1, min(6, 30//w)+1):
                    if mh*mw < 3:
                        continue
                    if regime == "scale":
                        valid = max(h, w) <= 6 if split == "train" else max(h, w) == 7
                    elif regime == "extent":
                        valid = max(mh, mw) <= 4 if split == "train" else max(mh, mw) >= 5
                    else:
                        direction = 1 if regime == "wide" else -1
                        valid = direction*(mw-mh) <= 0 if split == "train" else direction*(mw-mh) > 0
                    if valid:
                        choices.append((h, w, mh, mw))
    return choices


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = _pick(rng, ("scale", "extent", "wide", "tall"))
    consistent = rng.random() < 0.20
    base_palette = list(map(int, rng.choice(10, 7, replace=False)))
    puzzle, used = {"train": [], "test": []}, set()
    first_dims = first_anchor = None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        options = _options(regime, split)
        for index in range(count):
            teaching = split == "train" and index == 0
            if teaching:
                candidates = [v for v in options if v[2:] == (4, 4)]
            elif split == "train" and index == 1:
                candidates = [v for v in options if min(v[2:]) >= 2
                              and v[0] != first_dims[0] and v[1] != first_dims[1]]
            else:
                candidates = options
            for _ in range(24):
                h, w, mh, mw = _pick(rng, candidates)
                previous = first_anchor if split == "train" and index == 1 else None
                layout = _layout(rng, (mh, mw), teaching, previous)
                points = list(map(tuple, np.argwhere(layout).tolist()))
                anchors = [p for p in points if not teaching or p != ((mh-1)//2, (mw-1)//2)]
                if split == "train" and index == 1:
                    anchors = [p for p in anchors if p[0] != first_anchor[0] and p[1] != first_anchor[1]]
                assert anchors
                anchor = _pick(rng, anchors)
                degrees = _degrees(points)
                holes = {k: _cavities(rng, h, w, k, teaching) for k in sorted(set(degrees.values()))}
                palette = base_palette if consistent else list(map(int, rng.choice(10, 7, replace=False)))
                colors = (palette[0], palette[1], {k: palette[k+2] for k in holes})
                pair = _place(rng, layout, anchor, holes, colors)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    if teaching:
                        first_dims, first_anchor = (h, w), anchor
                    break
            else:
                # At most 13 previous inputs; 90 mappings of two visible colors
                # guarantee a novel input without an unbounded retry loop.
                present = list(map(int, np.unique(pair["input"])))
                domain = present[:2] + [v for v in range(10) if v not in present[:2]]
                found = False
                for a in range(10):
                    for b in range(10):
                        if a == b:
                            continue
                        lut = np.empty(10, dtype=np.int8)
                        lut[domain] = [a, b] + [v for v in range(10) if v not in (a, b)]
                        candidate = {k: lut[v] for k, v in pair.items()}
                        key = (candidate["input"].shape, candidate["input"].tobytes())
                        if key not in used:
                            used.add(key)
                            puzzle[split].append(candidate)
                            found = True
                            break
                    if found:
                        break
                assert found
    # Training order is irrelevant, including the position of the witness.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain only small feature sets and hashes, no files."""
    features = {k: set() for k in ("puzzles", "canvas", "glyph_dims", "map_dims", "layouts",
                                  "masks", "anchors", "areas", "counts", "palettes")}
    totals = {k: 0 for k in ("pairs", "disconnected", "gap_rows", "gap_cols", "overlap",
                             "lines", "asymmetric", "large_cavities")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        features["puzzles"].add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                m = _decode(inp)
                totals["pairs"] += 1
                for key, value in (("canvas", inp.shape), ("glyph_dims", (m["h"], m["w"])),
                                   ("map_dims", m["map_shape"]), ("layouts", m["points"]),
                                   ("anchors", m["anchor"]), ("counts", tuple(sorted(m["catalog"]))),
                                   ("palettes", tuple(map(int, np.unique(inp))))):
                    features[key].add(value)
                mask = np.zeros(m["map_shape"], dtype=bool)
                for p in m["points"]:
                    mask[p] = True
                totals["disconnected"] += len(_components(mask)) > 1
                totals["gap_rows"] += bool(np.any(~mask.any(axis=1)))
                totals["gap_cols"] += bool(np.any(~mask.any(axis=0)))
                totals["lines"] += min(mask.shape) == 1
                pr, pc = m["map_box"][:2]
                totals["overlap"] += any(out[pr+2*r, pc+2*c] != m["bg"] for r, c in m["points"])
                for k, e in m["catalog"].items():
                    features["masks"].add((e["holes"].shape, e["holes"].tobytes()))
                    totals["asymmetric"] += _asymmetric(e["holes"])
                    for comp in _components(e["holes"]):
                        features["areas"].add(len(comp))
                        totals["large_cavities"] += len(comp) >= 9
                assert np.array_equal(solve(inp.copy()), out)
                assert np.array_equal(solve(inp.T.copy()), out.T)
                lut = np.roll(np.arange(10, dtype=np.int8), seed % 9 + 1)
                assert np.array_equal(solve(lut[inp]), lut[out])
    minima = {"puzzles": 42, "canvas": 30, "glyph_dims": 9, "map_dims": 15,
              "layouts": 100, "masks": 100, "anchors": 12, "areas": 12,
              "counts": 10, "palettes": 30}
    assert all(len(features[k]) >= n for k, n in minima.items()), {k: len(v) for k, v in features.items()}
    assert all(totals[k] >= 3 for k in totals if k != "pairs"), totals
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    # One new map point changes existing neighbors' selected glyphs, while
    # unchanged distant sites keep their exact blocks. Registration is fixed.
    interaction_checked = False
    for pair in generate_puzzle(0, 2, 1)["train"]:
        m = _decode(pair["input"])
        occupied = set(m["points"])
        if len(m["catalog"]) != 5:
            continue
        for r in range(m["map_shape"][0]):
            for c in range(m["map_shape"][1]):
                q = (r, c)
                if q in occupied:
                    continue
                degrees = _degrees(occupied | {q})
                changed = [p for p in occupied if degrees[p] != m["degrees"][p]]
                if (not changed or set(degrees.values()) != set(m["catalog"])
                        or degrees[m["anchor"]] != m["anchor_degree"]):
                    continue
                altered = pair["input"].copy()
                pr, pc = m["map_box"][:2]
                altered[pr+2*r, pc+2*c] = m["marker"]
                prediction = solve(altered)
                after = _decode(altered)
                assert (after["top"], after["left"]) == (m["top"], m["left"])
                for p in occupied:
                    y, x = m["top"]+p[0]*m["h"], m["left"]+p[1]*m["w"]
                    sl = np.s_[y:y+m["h"], x:x+m["w"]]
                    assert np.array_equal(prediction[sl], pair["output"][sl]) == (p not in changed)
                interaction_checked = True
                break
            if interaction_checked:
                break
    assert interaction_checked
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1234+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "features": {k: len(v) for k, v in features.items()},
            "totals": totals, "local_interaction_checked": interaction_checked}
