"""The saved topology/empty-neighborhood composition, with input-only colors.

Fixed introduced colors: 2 = exterior halo or locally empty background,
8 = cavity-bearing body, 6 = cavity rim. Original-background highlighting
is applied LAST, so a sufficiently wide cavity has a 2 core inside its 6 rim.
No parent modules, files, latent palette, or generation state are used by solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


HIGHLIGHT, HOLLOW, FILL = 2, 8, 6
INKS = (0, 1, 3, 4, 5, 7, 9)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _palette(a):
    values, counts = np.unique(a, return_counts=True)
    if len(values) != 2 or counts.max() <= a.size / 2:
        raise ValueError("input must have two colors and a majority background")
    bg, fg = int(values[np.argmax(counts)]), int(values[np.argmin(counts)])
    if bg not in INKS or fg not in INKS:
        raise ValueError("input colors must exclude the three fixed output colors")
    return bg, fg


def _regions(mask, neighbors=N4):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        cells, stack = [], [(r, c)]
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in neighbors:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        groups.append(cells)
    return groups


def _near(mask):
    """Original-input closed king-move neighborhood, clipped to the canvas."""
    h, w = mask.shape
    p = np.pad(mask, 1, constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            result |= p[dr:dr + h, dc:dc + w]
    return result


def _analyze(a):
    bg, fg = _palette(a)
    body = a == fg
    labels = np.full(a.shape, -1, dtype=int)
    objects = _regions(body)
    for i, cells in enumerate(objects):
        rr, cc = np.asarray(cells).T
        labels[rr, cc] = i
    holes = np.zeros_like(body)
    hollow_ids = set()
    h, w = a.shape
    for cells in _regions(~body):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        owners = {int(labels[r + dr, c + dc]) for r, c in cells for dr, dc in N4
                  if labels[r + dr, c + dc] >= 0}
        if len(owners) != 1:
            raise ValueError("collective or nested enclosure is outside this family")
        hollow_ids.update(owners)
        rr, cc = np.asarray(cells).T
        holes[rr, cc] = True
    hollow = np.isin(labels, list(hollow_ids))
    near = _near(body)
    return {"background": bg, "foreground": fg, "body": body, "labels": labels,
            "objects": objects, "holes": holes, "hollow": hollow,
            "solid": body & ~hollow, "halo": ~body & ~holes & near,
            "deep": ~near, "exterior": ~body & ~holes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Classify ORIGINAL objects, decorate, then overlay original local emptiness."""
    a = _grid(input_grid)
    z = _analyze(a)
    out = a.copy()
    out[z["hollow"]] = HOLLOW
    out[z["holes"]] = FILL
    out[z["halo"]] = HIGHLIGHT
    # This includes empty 3x3 centers INSIDE sealed holes. Never test out here.
    out[z["deep"]] = HIGHLIGHT
    return out


def _local_holes(mask, neighbors=N4):
    """Object-local audit/construction helper, explicitly padded with outside."""
    p = np.pad(mask, 1, constant_values=False)
    holes = np.zeros_like(p)
    for cells in _regions(~p, neighbors):
        if (0, 0) not in cells:
            rr, cc = np.asarray(cells).T
            holes[rr, cc] = True
    return holes[1:-1, 1:-1]


def _unambiguous(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _turn(mask, rotation, flip):
    mask = np.rot90(mask, rotation)
    return _crop(np.fliplr(mask) if flip else mask)


def _orient(rng, mask):
    return _turn(mask, int(rng.integers(4)), bool(rng.integers(2)))


def _lobe(rng, mask):
    """Add a solid spur outside a full boundary; do not change existing holes."""
    k = int(rng.integers(4))
    mask = np.rot90(mask, k)
    h, w = mask.shape
    length = int(rng.integers(1, 4))
    thickness = int(rng.integers(1, min(w, 3) + 1))
    start = int(rng.integers(0, w - thickness + 1))
    out = np.pad(mask, ((0, length), (0, 0)))
    out[h:, start:start + thickness] = True
    # _lobe is only called on bodies with full rectangular outer boundaries.
    return np.rot90(out, -k)


def _contrast(rng):
    # Exactly a 3x3 pocket: its single deep center defeats A-v2's >=5 cutoff.
    top, bottom, left, right = map(int, rng.integers(1, 3, size=4))
    h, w = 3 + top + bottom, 3 + left + right
    closed = np.ones((h, w), bool)
    closed[top:top + 3, left:left + 3] = False
    # Add the same bottom lobe to both versions, away from the top door.
    tail = int(rng.integers(1, 4))
    start = int(rng.integers(0, w - 1))
    closed = np.pad(closed, ((0, tail), (0, 0)))
    closed[h:, start:start + 2] = True
    opened = closed.copy()
    opened[:top, left + 1] = False
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    return [_turn(m, k, flip) for m in (closed, opened)]


def _chamber(rng, kind="wide"):
    if kind == "tiny":
        ph, pw = 1, int(rng.integers(1, 4))
    else:
        ph, pw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    top, bottom, left, right = map(int, rng.integers(1, 3, size=4))
    m = np.ones((ph + top + bottom, pw + left + right), bool)
    cavity = np.ones((ph, pw), bool)
    if kind == "irregular":
        cavity[0, -1] = False
        if ph >= 4 and pw >= 4 and rng.random() < .5:
            cavity[0, -2] = False
    m[top:top + ph, left:left + pw] &= ~cavity
    if rng.random() < .6:
        m = _lobe(rng, m)
    return _orient(rng, m)


def _multi(rng):
    # One narrow pocket and one irregular, wide pocket in the SAME body.
    ph, pw = int(rng.integers(3, 5)), int(rng.integers(3, 5))
    small_w = int(rng.integers(1, 3))
    wall = int(rng.integers(1, 3))
    sep = int(rng.integers(1, 3))
    h, w = ph + 2 * wall, small_w + sep + pw + 2 * wall
    m = np.ones((h, w), bool)
    m[wall:wall + int(rng.integers(1, 3)), wall:wall + small_w] = False
    x = wall + small_w + sep
    m[wall:wall + ph, x:x + pw] = False
    m[wall, x + pw - 1] = True
    if rng.random() < .5:
        m = _lobe(rng, m)
    return _orient(rng, m)


def _solid(rng, small=False):
    limit = 5 if small else 8
    h, w = int(rng.integers(1, limit)), int(rng.integers(1, limit))
    style = int(rng.integers(4))
    if style == 0:
        m = np.ones((1, int(rng.integers(1, limit + 1))), bool)
    elif style == 1 and min(h, w) >= 3:
        m = np.ones((h, w), bool)
        m[:h - 1, 1:w - 1] = False  # deep open U, with separated runs
    elif style == 2:
        # A variable-width staircase mass: connected, with no sealed pocket.
        m = np.zeros((h, w), bool)
        widths = np.sort(rng.integers(1, w + 1, size=h))
        for r, width in enumerate(widths):
            m[r, :width] = True
    else:
        m = np.ones((h, w), bool)
    return _orient(rng, m)


def _mask(rng, kind, small=False):
    if kind == "solid":
        return _solid(rng, small)
    if kind == "open":
        return _contrast(rng)[1]
    if kind == "multi":
        return _multi(rng)
    return _chamber(rng, kind)


def _pack(rng, masks, h, w, margin, gap):
    # Boxes keep topology ownership clear. Positions have no fixed slots.
    order = sorted(range(len(masks)), key=lambda i: -masks[i].size)
    for _ in range(8):
        canvas = np.zeros((h, w), bool)
        occupied = np.zeros_like(canvas)
        for i in order:
            m = masks[i]
            mh, mw = m.shape
            choices = [(r, c) for r in range(margin, h - mh - margin + 1)
                       for c in range(margin, w - mw - margin + 1)
                       if not occupied[r:r + mh, c:c + mw].any()]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            canvas[r:r + mh, c:c + mw] |= m
            occupied[max(0, r - gap):min(h, r + mh + gap),
                     max(0, c - gap):min(w, c + mw + gap)] = True
        else:
            if canvas.mean() < .48:
                return canvas
    return None


def _scene(rng, required, kinds):
    masks = required + [_mask(rng, k) for k in kinds]
    margin, gap = int(rng.integers(0, 4)), int(rng.integers(1, 4))
    area = sum(m.size for m in masks)
    aspect = float(np.exp(rng.uniform(-.65, .65)))
    target = area / float(rng.uniform(.22, .42))
    h = max(10, int(np.sqrt(target * aspect)), max(m.shape[0] for m in masks) + 2 * margin)
    w = max(10, int(np.sqrt(target / aspect)), max(m.shape[1] for m in masks) + 2 * margin)
    for growth in (0, 5, 30):
        canvas = _pack(rng, masks, min(30, h + growth), min(30, w + growth), margin, gap)
        if canvas is not None:
            return canvas
    # Bound dense packing work: preserve teaching bodies; simplify extras.
    simple = []
    for kind in kinds:
        if kind in ("solid", "open"):
            m = np.ones((1, int(rng.integers(1, 4))), bool)
        else:
            m = np.ones((3, 3), bool)
            m[1, 1] = False
        simple.append(m)
    masks = required + simple
    canvas = _pack(rng, masks, 30, 30, 0, 1)
    if canvas is not None:
        return canvas
    # Guaranteed finite shelf fallback: <=2 required <=13-cell masks above,
    # <=6 tiny extras below. It is a packing fallback, never a solving rule.
    canvas = np.zeros((30, 30), bool)
    c, bottom = 0, 0
    for m in required:
        mh, mw = m.shape
        assert c + mw <= 30
        canvas[:mh, c:c + mw] = m
        c += mw + 1
        bottom = max(bottom, mh)
    for i, m in enumerate(simple):
        r, c = bottom + 2 + 5 * (i // 5), 6 * (i % 5)
        canvas[r:r + m.shape[0], c:c + m.shape[1]] = m
    return canvas


def _integer(name, value, lo, hi=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < lo or (hi is not None and value > hi)):
        raise ValueError(name + " is outside its integer range")
    return int(value)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(INKS)))
    vary_palette = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if split == "train" and index == 0:
                required = _contrast(rng)
                n = int(rng.integers(2, 5))
            elif split == "train" and index == 1:
                required = [_multi(rng), _solid(rng, True)]
                n = int(rng.integers(2, 5))
            else:
                # The test shift is a new count, using already-taught shapes.
                n = int(rng.integers(5, 8) if split == "test" else rng.integers(2, 5))
                required = [_chamber(rng, str(rng.choice(["wide", "irregular"]))),
                            _solid(rng)]
            kinds = [str(rng.choice(["solid", "solid", "open", "tiny", "wide", "irregular", "multi"]))
                     for _ in range(n - len(required))]
            if vary_palette:
                bg, fg = map(int, rng.choice(INKS, size=2, replace=False))
                if split == "train" and index == 1:
                    bg, fg = palette[2:4]
            else:
                bg, fg = palette[:2]
            for _ in range(24):
                body = _scene(rng, required, kinds)
                inp = np.where(body, fg, bg).astype(np.int8)
                if _key(inp) not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": solve(inp)})
    return puzzle


def _separated_runs(mask):
    return any(np.any(np.diff(np.flatnonzero(row)) > 1) for rows in (mask, mask.T) for row in rows)


def _irregular(cells):
    rr, cc = np.asarray(cells).T
    return len(cells) < (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)


def _audit_pair(pair):
    """Pixel invariants, object-local topology, and direct window predicates.

    This is a generator-owned audit, not an external reference solver.
    Unlike solve's global cavity ownership, audit holes one cropped object
    at a time and compare pointwise expectations with the returned output.
    """
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == out.shape
    bg, fg = _palette(a)
    body = a == fg
    assert _unambiguous(body), "diagonal leak/contact"
    objects = _regions(body)
    assert len(objects) == len(_regions(body, N8)), "diagonal object contact"
    local_holes, hollow, solids = np.zeros_like(body), np.zeros_like(body), np.zeros_like(body)
    records = []
    for cells in objects:
        rr, cc = np.asarray(cells).T
        r, c = int(rr.min()), int(cc.min())
        m = np.zeros((int(rr.max()) - r + 1, int(cc.max()) - c + 1), bool)
        m[rr - r, cc - c] = True
        holes = _local_holes(m)
        assert np.array_equal(holes, _local_holes(m, N8)), "ambiguous cavity"
        hr, hc = np.nonzero(holes)
        assert not body[hr + r, hc + c].any(), "nested object"
        assert not local_holes[hr + r, hc + c].any(), "overlapping ownership"
        local_holes[hr + r, hc + c] = True
        (hollow if holes.any() else solids)[rr, cc] = True
        records.append((m, holes))
    assert np.array_equal(local_holes, _local_holes(body)), "collective enclosure"
    assert np.all(out[hollow] == HOLLOW), "whole hollow body must change"
    assert np.all(out[solids] == fg), "cavity-free body must retain ink"
    deep = np.zeros_like(body)
    h, w = a.shape
    for r, c in np.argwhere(~body):
        r, c = int(r), int(c)
        empty = not body[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)].any()
        deep[r, c] = empty
        expected = FILL if local_holes[r, c] and not empty else HIGHLIGHT
        assert out[r, c] == expected, "background priority/locality violation"
    exterior = ~body & ~local_holes
    halo = exterior & ~deep
    assert not (halo & deep).any()
    assert np.all(out[exterior] == HIGHLIGHT), "halo plus deep must cover the exterior"
    assert np.any(deep & exterior) and halo.any(), "both A and B must paint exterior cells"
    assert not np.any(out == bg), "the merged rule paints all original background"
    return {"objects": records, "solid": solids, "hollow": hollow, "holes": local_holes,
            "deep": deep, "halo": halo, "exterior": exterior, "foreground": fg}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, counts = set(), [], {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            z = _audit_pair(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            counts[split].append(len(z["objects"]))
            if split == "train":
                training.append(z)
    assert min(counts["test"]) > max(counts["train"]), "missing count extrapolation"
    assert any(z["solid"].any() and z["hollow"].any() for z in training)
    assert any(_separated_runs(m) for z in training for m, holes in z["objects"] if not holes.any())
    assert any(len(_regions(holes)) >= 2 for z in training for m, holes in z["objects"])
    assert any(_irregular(cells)
               for z in training for m, holes in z["objects"] for cells in _regions(holes)), "no irregular hole"
    # Tiny local candidate components are highlighted, without parent's >=5 filter.
    assert any(len(cells) == 1 for z in training for cells in _regions(z["deep"] & z["holes"]))
    assert any(not any(z["deep"][r, c] for r, c in cells)
               for z in training for cells in _regions(z["holes"])), "no narrow cavity demonstration"
    assert all(any(z[role].any() for z in training) for role in ("solid", "hollow", "holes", "halo", "deep"))


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _regressions():
    # Fixed expectations exercise priority, clipping, tiny cores, open bays,
    # and recoloring an attached tail as part of the entire hollow object.
    a = np.zeros((9, 15), dtype=np.int8)
    a[1:6, 1:6] = 1
    a[2:5, 2:5] = 0
    a[5:8, 1] = 1
    a[1:6, 9:14] = 1
    a[2:5, 10:13] = 0
    a[1, 11] = 0
    b = solve(a)
    assert b[3, 3] == HIGHLIGHT and b[2, 2] == FILL
    assert b[7, 1] == HOLLOW and b[5, 9] == 1
    assert b[3, 11] == HIGHLIGHT and b[0, 0] == HIGHLIGHT
    assert b[1, 11] == HIGHLIGHT  # the halo can close a door, never reclassify it
    # Radius-one vs radius-two, and center-only vs union-of-empty-windows.
    assert b[2, 3] == FILL  # part of the empty 3x3 window, but not its center
    # A corner object clips its halo without wrapping to the opposite edge.
    c = np.zeros((6, 7), dtype=np.int8)
    c[0, 0] = 1
    assert solve(c)[0, 0] == 1 and solve(c)[0, 1] == HIGHLIGHT
    # Input-only result, even when no stored generated example has this size.
    assert np.array_equal(solve([[0, 0, 1]]), [[2, 2, 1]])


def self_test():
    """Stream 42 seeds; no files, retained puzzles, execution traces or logs."""
    _regressions()
    signatures, layouts, shapes, silhouettes, palettes, combos = set(), set(), set(), set(), set(), set()
    features = {"irregular_holes": 0, "multiple_holes": 0, "open_bays": 0,
                "tiny_objects": 0, "clipped": 0, "wide_cavity_cores": 0}
    pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        for split in ("train", "test"):
            for pair in p[split]:
                pairs += 1
                a = pair["input"]
                z = _audit_pair(pair)
                original = a.copy()
                assert np.array_equal(solve(a), pair["output"]) and np.array_equal(a, original)
                bg, fg = _palette(a)
                palettes.add((bg, fg))
                shapes.add(a.shape)
                body = a == fg
                layouts.add((a.shape, hashlib.sha256(body.tobytes()).hexdigest()))
                combos.add((len(z["objects"]), sum(holes.any() for m, holes in z["objects"])))
                features["clipped"] += bool(body[0].any() or body[-1].any() or body[:, 0].any() or body[:, -1].any())
                features["wide_cavity_cores"] += bool((z["deep"] & z["holes"]).any())
                for m, holes in z["objects"]:
                    silhouettes.add(_key(m))
                    features["open_bays"] += not holes.any() and _separated_runs(m)
                    features["multiple_holes"] += len(_regions(holes)) > 1
                    features["tiny_objects"] += m.sum() <= 3
                    for cells in _regions(holes):
                        features["irregular_holes"] += _irregular(cells)
                # Explicit necessity ablations on original-input zones.
                ablate_a = a.copy()
                ablate_a[z["hollow"]] = HOLLOW
                ablate_a[z["holes"]] = FILL
                ablate_a[z["halo"]] = HIGHLIGHT
                assert not np.array_equal(ablate_a, pair["output"])
                ablate_b = a.copy()
                ablate_b[z["deep"]] = HIGHLIGHT
                assert not np.array_equal(ablate_b, pair["output"])
    assert len(signatures) == 42 and len(layouts) >= 200
    assert len(shapes) >= 35 and len(silhouettes) >= 100 and len(palettes) >= 15
    assert len(combos) >= 10 and all(value >= 3 for value in features.values())
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(730 + nt + nq, nt, nq)
        validate_generated_puzzle(p)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corruption checks target each distinguishable output class.
    for role in ("solid", "hollow", "holes", "deep", "halo"):
        p = generate_puzzle(12, 2, 1)
        pair = p["train"][0]
        z = _audit_pair(pair)
        r, c = np.argwhere(z[role])[0]
        pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted corrupted " + role)
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "binary_layouts": len(layouts),
            "object_silhouettes": len(silhouettes), "input_palettes": len(palettes),
            "count_class_combinations": len(combos),
            "features": {k: int(v) for k, v in features.items()},
            "verification_scope": "generator-owned invariants and fixed regressions only"}
