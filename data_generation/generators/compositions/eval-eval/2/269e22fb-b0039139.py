"""Count-addressed crop registration. All solving information is in the input.

Two framed, color-keyed cards contain a reference and an exact query fragment.
Loose objects in the query's primary color select the nth D4-matching rectangle.
Only that rectangle is recolored; the output retains the reference's orientation.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _transform(a, k):
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4).copy()


def _canonical(a):
    return min(_key(_transform(a, k)) for k in range(8))


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    group.append(point)
        groups.append(group)
    return groups


def _tight(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty mask")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _boxes(reference, query, indices=range(8)):
    """Match exact binary windows, deduplicating footprints, not orientations."""
    result, seen = set(), set()
    for k in indices:
        patch = _transform(query, k)
        key = _key(patch)
        if key in seen:
            continue
        seen.add(key)
        h, w = patch.shape
        if h > reference.shape[0] or w > reference.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(reference, (h, w))
        rows, cols = np.nonzero(np.all(windows == patch, axis=(-2, -1)))
        result.update((int(r), int(c), h, w) for r, c in zip(rows, cols))
    return sorted(result)


def _unambiguous_anchors(boxes):
    return len({(r, c) for r, c, _, _ in boxes}) == len(boxes)


def _read_card(g, component, frame):
    points = np.asarray(component)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    r, c, bottom, right = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
    a = g[r:bottom, c:right]
    if a.shape[0] < 7 or a.shape[1] < 5:
        return None
    if not (np.all(a[[0, -1], :] == frame) and np.all(a[:, [0, -1]] == frame)
            and np.all(a[2] == frame)):
        return None
    header = a[1, 1:-1]
    cuts = np.flatnonzero(header == frame)
    if len(cuts) != 1 or not 0 < cuts[0] < len(header) - 1:
        return None
    cut = int(cuts[0])
    primary, secondary = int(header[0]), int(header[-1])
    if primary == secondary or not (np.all(header[:cut] == primary)
                                    and np.all(header[cut + 1:] == secondary)):
        return None
    payload = a[3:-1, 1:-1]
    if set(map(int, np.unique(payload))) != {primary, secondary}:
        return None
    # Every frame-colored cell in this bounding box belongs to this component.
    if int(np.count_nonzero(a == frame)) != len(component):
        return None
    return {"bounds": (r, c, bottom, right), "payload": payload.copy(),
            "binary": payload == primary, "primary": primary,
            "secondary": secondary, "swatch_widths": (cut, len(header) - cut - 1)}


def _decode(input_grid):
    g = _array(input_grid)
    bg = int(g[0, 0])
    _require(np.all(g[[0, -1]] == bg) and np.all(g[:, [0, -1]] == bg),
             "input needs a blank outer margin")
    candidates = []
    for frame in map(int, np.unique(g)):
        if frame == bg:
            continue
        groups = _components(g == frame)
        if len(groups) != 2 or len(_components(g == frame, True)) != 2:
            continue
        cards = [_read_card(g, group, frame) for group in groups]
        if any(card is None for card in cards):
            continue
        cards.sort(key=lambda card: card["payload"].size, reverse=True)
        ref, query = cards
        if ref["payload"].size <= query["payload"].size:
            continue
        roles = (bg, frame, ref["primary"], ref["secondary"],
                 query["primary"], query["secondary"])
        if len(set(roles)) != 6:
            continue
        outside = np.ones(g.shape, dtype=bool)
        guard = np.zeros(g.shape, dtype=bool)
        for card in cards:
            r, c, b, e = card["bounds"]
            if not outside[r:b, c:e].all():
                break
            outside[r:b, c:e] = False
            guard[max(0, r-1):min(g.shape[0], b+1), max(0, c-1):min(g.shape[1], e+1)] = True
        else:
            if not np.all(np.isin(g[outside], [bg, query["primary"]])):
                continue
            tokens = outside & (g == query["primary"])
            if np.any(tokens & guard):
                continue
            objects = _components(tokens)
            if not objects or len(objects) != len(_components(tokens, True)):
                continue
            boxes = _boxes(ref["binary"], query["binary"])
            n = len(objects)
            if n > len(boxes) or not _unambiguous_anchors(boxes):
                continue
            candidates.append({"grid": g, "roles": roles, "reference": ref,
                               "query": query, "tokens": tokens, "objects": objects,
                               "count": n, "matches": boxes, "selected": boxes[n-1]})
    _require(len(candidates) == 1, "input has no unique valid card/counter interpretation")
    return candidates[0]


def _paint(reference, binary, box, primary, secondary):
    out = reference.copy()
    r, c, h, w = box
    out[r:r+h, c:c+w] = np.where(binary[r:r+h, c:c+w], primary, secondary)
    return out.astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pure input-only solution; no learned bitmap, seed, labels or cache."""
    d = _decode(input_grid)
    ref, q = d["reference"], d["query"]
    return _paint(ref["payload"], ref["binary"], d["selected"], q["primary"], q["secondary"])


def _grow(h, w, area, rng):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < h and 0 <= y < w and not mask[x, y]:
                frontier.add((x, y))
    return mask


def _motif(rng):
    for _ in range(80):
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        mode = int(rng.integers(3))
        mask = np.zeros((h, w), dtype=bool)
        if mode == 0:  # Connected branches joining independently chosen sides.
            pivot = (int(rng.integers(h)), int(rng.integers(w)))
            for r, c in ((0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                         (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)):
                mask[r, c] = True
                while (r, c) != pivot:
                    moves = []
                    if r != pivot[0]:
                        moves.append((r + (1 if r < pivot[0] else -1), c))
                    if c != pivot[1]:
                        moves.append((r, c + (1 if c < pivot[1] else -1)))
                    r, c = moves[int(rng.integers(len(moves)))]
                    mask[r, c] = True
        elif mode == 1:  # Unequal loop with a notch/branch.
            mask[[0, -1], :] = True
            mask[:, [0, -1]] = True
            for _ in range(int(rng.integers(1, h+w))):
                mask[int(rng.integers(h)), int(rng.integers(w))] ^= True
        else:  # Dense connected silhouette, pruned without shrinking its box.
            mask[:] = True
            for index in rng.permutation(h*w)[:int(rng.integers(2, h*w//2+1))]:
                r, c = divmod(int(index), w)
                mask[r, c] = False
                if len(_components(mask)) != 1 or _tight(mask).shape != (h, w):
                    mask[r, c] = True
        if (mask.any() and not mask.all() and _tight(mask).shape == mask.shape
                and len({_key(_transform(mask, k)) for k in range(8)}) == 8):
            # Retaining a blank edge is essential: an exact crop is not a tight
            # foreground bounding box. This row participates in the matching.
            return np.pad(mask, ((1, 0), (0, 0)))
    return np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 1, 0]], dtype=bool)


def _packing_options(a, b):
    ah, aw = a.shape[0]+4, a.shape[1]+2
    bh, bw = b.shape[0]+4, b.shape[1]+2
    options = []
    if max(ah, bh)+2 <= 30 and aw+bw+3 <= 30:
        options.append(False)  # side by side
    if ah+bh+3 <= 30 and max(aw, bw)+2 <= 30:
        options.append(True)  # stacked
    return options


def _reference(rng):
    for attempt in range(120):
        query = _motif(rng)
        nr, nc = ((2, 2), (2, 3), (3, 2))[int(rng.integers(3))]
        if attempt == 119:
            query = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 1, 0]], dtype=bool)
            nr = nc = 2
        core = query[1:]
        s = max(core.shape)
        rows = [s + int(rng.integers(2)) for _ in range(nr)]
        cols = [s + int(rng.integers(2)) for _ in range(nc)]
        if attempt == 119:
            rows, cols = [s]*nr, [s]*nc
        H, W = sum(rows)+nr+1, sum(cols)+nc+1
        ref = np.zeros((H, W), dtype=bool)
        if not all(_packing_options(ref, _transform(query, k)) for k in range(8)):
            continue
        protected = np.zeros_like(ref)
        orientations = list(map(int, rng.integers(0, 8, nr*nc)))
        orientations[1] = int(rng.integers(1, 4))
        orientations[2] = int(rng.integers(4, 8))
        r0, j = 1, 0
        for height in rows:
            c0 = 1
            for width in cols:
                k = orientations[j]
                tile = _transform(core, k)
                h, w = tile.shape
                r, c = r0+int(rng.integers(height-h+1)), c0+int(rng.integers(width-w+1))
                ref[r:r+h, c:c+w] = tile
                protected[max(0, r-1):min(H, r+h+1), max(0, c-1):min(W, c+w+1)] = True
                c0 += width+1
                j += 1
            r0 += height+1
        # Context marks occupy unused room, never the guard around a repeated
        # core. They are part of the reference, not tokens or extra instructions.
        for _ in range(int(rng.integers(0, 5))):
            free = np.argwhere(~protected)
            if not len(free):
                break
            r, c = free[int(rng.integers(len(free)))]
            length = int(rng.integers(1, 4))
            for y in range(int(c), min(W, int(c)+length)):
                if not protected[r, y]:
                    ref[r, y] = True
        boxes = _boxes(ref, query)
        if len(boxes) != nr*nc or not _unambiguous_anchors(boxes):
            continue
        # These are evidence constraints, not part of the solution's rule.
        if boxes[1] not in _boxes(ref, query, range(1, 4)):
            continue
        if boxes[2] in _boxes(ref, query, range(4)):
            continue
        column_order = sorted(boxes, key=lambda b: (b[1], b[0]))
        if column_order[1] == boxes[1] and column_order[2] == boxes[2]:
            continue
        if len({_key(_transform(ref, k)) for k in range(8)}) != 8:
            continue
        return ref, query
    raise RuntimeError("bounded reference construction exhausted")


def _card(binary, primary, secondary, frame, rng, avoid_width=None):
    h, w = binary.shape
    a = np.full((h+4, w+2), frame, dtype=np.int8)
    cuts = [cut for cut in range(1, w-1)
            if cut != avoid_width and w-cut-1 != avoid_width]
    # For a width-four query, one swatch is necessarily width two. The
    # count-three witness can always avoid its rank in both swatches.
    if not cuts:
        cuts = list(range(1, w-1))
    cut = int(rng.choice(cuts))
    a[1, 1:1+cut] = primary
    a[1, 2+cut:-1] = secondary
    a[3:-1, 1:-1] = np.where(binary, primary, secondary)
    return a


def _token_shape(area, rng):
    mode = int(rng.integers(4))
    if mode == 0 and area <= 8:
        mask = np.ones((1, area), dtype=bool)
        return mask.T if rng.integers(2) else mask
    if mode == 1:
        divisors = [d for d in range(1, area+1) if area % d == 0 and max(d, area//d) <= 6]
        if divisors:
            h = int(rng.choice(divisors))
            return np.ones((h, area//h), dtype=bool)
    if mode == 2 and area == 8:
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    side = max(2, int(np.ceil(np.sqrt(area)))+1)
    return _tight(_grow(side, side, area, rng))


def _place_input(ref, query, palette, n, area, rng):
    bg, frame, rp, rs, qp, qs = palette
    avoid = n if area is not None else None
    rc = _card(ref, rp, rs, frame, rng, avoid)
    qc = _card(query, qp, qs, frame, rng, avoid)
    options = _packing_options(ref, query)
    _require(options, "cards do not fit")
    sizes = [1]*n
    if area is None:
        sizes = list(map(int, rng.integers(1, 10, n)))
    else:
        # Equal total area in the two mandatory witnesses, unequal counts.
        sizes = [2]*n
        for _ in range(area-2*n):
            sizes[int(rng.integers(n))] += 1
    for attempt in range(50):
        stacked = bool(rng.choice(options))
        cards = [rc, qc]
        if rng.integers(2):
            cards.reverse()
        a, b = cards
        if stacked:
            minH, minW = a.shape[0]+b.shape[0]+3, max(a.shape[1], b.shape[1])+2
        else:
            minH, minW = max(a.shape[0], b.shape[0])+2, a.shape[1]+b.shape[1]+3
        H = int(rng.integers(max(22, minH), 31)) if attempt < 40 else 30
        W = int(rng.integers(max(22, minW), 31)) if attempt < 40 else 30
        g = np.full((H, W), bg, dtype=np.int8)
        free = np.ones((H, W), dtype=bool)
        free[[0, -1]] = False
        free[:, [0, -1]] = False
        if stacked:
            ra = int(rng.integers(1, H-a.shape[0]-b.shape[0]-1))
            rb = int(rng.integers(ra+a.shape[0]+1, H-b.shape[0]))
            ca, cb = int(rng.integers(1, W-a.shape[1])), int(rng.integers(1, W-b.shape[1]))
        else:
            ca = int(rng.integers(1, W-a.shape[1]-b.shape[1]-1))
            cb = int(rng.integers(ca+a.shape[1]+1, W-b.shape[1]))
            ra, rb = int(rng.integers(1, H-a.shape[0])), int(rng.integers(1, H-b.shape[0]))
        for card, r, c in ((a, ra, ca), (b, rb, cb)):
            h, w = card.shape
            g[r:r+h, c:c+w] = card
            free[r-1:r+h+1, c-1:c+w+1] = False
        success = True
        for size in sorted(sizes, reverse=True):
            found = False
            for _ in range(20):
                shape = _token_shape(size, rng)
                h, w = shape.shape
                windows = np.lib.stride_tricks.sliding_window_view(free, (h, w))
                positions = np.argwhere(np.all(windows | ~shape, axis=(-2, -1)))
                if not len(positions):
                    continue
                r, c = positions[int(rng.integers(len(positions)))]
                region = g[r:r+h, c:c+w]
                region[shape] = qp
                for x, y in np.argwhere(shape):
                    x, y = int(r+x), int(c+y)
                    free[max(0, x-1):x+2, max(0, y-1):y+2] = False
                found = True
                break
            if not found:
                success = False
                break
        if success:
            return g
    raise RuntimeError("bounded token packing exhausted")


def _query_variants(full, reference, test):
    H, W = full.shape
    variants = {}
    for h in range(3, H+1):
        for w in range(3, W+1):
            # Test fragments are strictly smaller than every training query.
            if (test and h*w > .75*full.size) or (not test and h*w < .85*full.size):
                continue
            for r in range(H-h+1):
                for c in range(W-w+1):
                    q = full[r:r+h, c:c+w].copy()
                    if q.all() or not q.any():
                        continue
                    boxes = _boxes(reference, q)
                    if len(boxes) >= 4 and _unambiguous_anchors(boxes):
                        variants[_key(q)] = q
    return list(variants.values())


def _count(value, name, lo, hi):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
             and lo <= int(value) <= hi, f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    for _ in range(30):
        reference, full = _reference(rng)
        test_variants = _query_variants(full, reference, True)
        if test_variants:
            break
    else:
        raise RuntimeError("could not construct an identifying smaller test fragment")
    train_variants = _query_variants(full, reference, False)
    palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    shift = int(rng.integers(1, 10))
    pool = [palette, tuple((c+shift) % 10 for c in palette)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for i in range(amount):
            witness = split == "train" and i < 2
            if witness:
                query, n, area = full.copy(), i+2, 12
                colors = pool[i]
            else:
                variants = test_variants if split == "test" else train_variants
                query = variants[int(rng.integers(len(variants)))].copy()
                query = _transform(query, int(rng.integers(8)))
                matches = _boxes(reference, query)
                n = int(rng.integers(1, min(8, len(matches))+1))
                area = None
                colors = (pool[int(rng.integers(2))] if rng.random() < .45
                          else tuple(map(int, rng.choice(10, 6, replace=False))))
            matches = _boxes(reference, query)
            for _ in range(20):
                inp = _place_input(reference, query, colors, n, area, rng)
                if _key(inp) not in seen:
                    break
            else:
                raise RuntimeError("duplicate input retry exhausted")
            # Output construction uses chosen visible roles and an exact box,
            # never solve(). The validator will parse the actual returned input.
            rp, rs, qp, qs = colors[2:]
            literal_ref = np.where(reference, rp, rs).astype(np.int8)
            out = _paint(literal_ref, reference, matches[n-1], qp, qs)
            puzzle[split].append({"input": inp, "output": out})
            seen.add(_key(inp))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad pair counts")
    seen, decoded = set(), {"train": [], "test": []}
    reference = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _array(pair["input"]), _array(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            ref, q = d["reference"], d["query"]
            if reference is None:
                reference = ref["binary"]
            _require(np.array_equal(reference, ref["binary"]), "reference differs within puzzle")
            _require(len(d["matches"]) >= 4, "generated queries need several genuine alternatives")
            _require(out.shape == ref["payload"].shape, "output must be the full reference canvas")
            r, c, h, w = d["selected"]
            changed = out != ref["payload"]
            footprint = np.zeros(out.shape, dtype=bool)
            footprint[r:r+h, c:c+w] = True
            _require(np.array_equal(changed, footprint), "only the entire selected rectangle changes")
            _require(np.array_equal(out[~footprint], ref["payload"][~footprint]), "context changed")
            _require(np.all(out[footprint] == np.where(ref["binary"][footprint],
                                                     q["primary"], q["secondary"])), "bad role recoloring")
            _require(np.array_equal(solve(inp), out), "input-only solution disagrees")
            decoded[split].append(d)
    train = decoded["train"]
    _require(any(a["count"] != b["count"] and a["tokens"].sum() == b["tokens"].sum()
                 and _canonical(a["query"]["binary"]) == _canonical(b["query"]["binary"])
                 for a in train for b in train), "missing equal-area/different-count witnesses")
    _require(any(d["count"] > 1 for d in train), "first-match shortcut not distinguished")
    _require(any(d["selected"] not in _boxes(reference, d["query"]["binary"], [0]) for d in train),
             "training does not require geometric matching")
    _require(any(d["selected"] not in _boxes(reference, d["query"]["binary"], range(4)) for d in train),
             "training does not demonstrate reflection")
    _require(any(_tight(d["query"]["binary"]).size < d["query"]["binary"].size for d in train),
             "missing exact-blank-boundary witness")
    _require(any(sorted(d["matches"], key=lambda b: (b[1], b[0]))[d["count"]-1] != d["selected"]
                 for d in train), "row/column ordering is not distinguished")
    _require(max(d["query"]["binary"].size for d in decoded["test"])
             < min(d["query"]["binary"].size for d in train), "missing smaller-fragment test shift")
    _require(len({d["roles"] for d in train}) >= 2, "missing color-role variation")
    for card in ("reference", "query"):
        for side in range(2):
            _require(any(d[card]["swatch_widths"][side] != d["count"] for d in train),
                     "swatch width could replace the counter")


def _signature(p):
    return tuple((s, _key(pair["input"]), _key(pair["output"]))
                 for s in ("train", "test") for pair in p[s])


def self_test():
    """Bounded grid-derived checks on 42 seeds and focused regressions."""
    corpus = []
    for seed in range(42):
        nt = 2 + seed % 9
        nq = 1 + seed % min(4, nt-1)
        corpus.append(generate_puzzle(seed, nt, nq))
    references, dimensions, query_shapes, counts, sizes, palettes = set(), set(), set(), set(), set(), set()
    selected, matched, layouts = set(), set(), set()
    for p in corpus:
        for split in ("train", "test"):
            for pair in p[split]:
                d = _decode(pair["input"])
                references.add(_canonical(d["reference"]["binary"]))
                dimensions.add(d["reference"]["binary"].shape)
                query_shapes.add(d["query"]["binary"].shape)
                counts.add(d["count"])
                sizes.update(map(len, d["objects"]))
                palettes.add(d["roles"])
                selected.add(d["selected"])
                matched.add(len(d["matches"]))
                layouts.add((pair["input"].shape, d["reference"]["bounds"], d["query"]["bounds"]))
    assert len({_signature(p) for p in corpus}) == 42
    assert len(references) >= 40, "reference structure collapses after D4 normalization"
    assert len(dimensions) >= 12 and len(query_shapes) >= 10
    assert 1 in counts and max(counts) >= 6 and len(sizes) >= 7
    assert len(palettes) >= 60 and len(selected) >= 40 and len(layouts) >= 100
    assert len(matched) >= 4
    for seed in (0, 17, 41):
        assert _signature(generate_puzzle(seed)) == _signature(generate_puzzle(seed))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(97, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A counter-only change moves the registration without changing either card.
    p = generate_puzzle(0, 2, 1)
    d = next(_decode(pair["input"]) for pair in p["train"] if _decode(pair["input"])["count"] == 3)
    modified = d["grid"].copy()
    for r, c in d["objects"][-1]:
        modified[r, c] = d["roles"][0]
    assert _decode(modified)["count"] == 2
    assert not np.array_equal(solve(modified), solve(d["grid"]))
    # Removing the blank edge changes exact footprints; tight cropping is wrong.
    assert any(_boxes(d["reference"]["binary"], _tight(d["query"]["binary"])) != d["matches"]
               for pair in p["train"] for d in [_decode(pair["input"])])
    # A symmetric query must not count several transforms of one box several times.
    probe = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    assert _boxes(probe, probe) == [(0, 0, 3, 3)]
    # Exercise solve() beyond sampled asymmetric cores: symmetric query,
    # boundary matches, and overlapping occurrences in a continuous field.
    rng = np.random.default_rng(731)
    for reference in (np.tile(probe, (2, 2)),
                      (np.indices((7, 7)).sum(axis=0) % 2 == 0)):
        matches = _boxes(reference, probe)
        assert len(matches) >= 4 and _unambiguous_anchors(matches)
        inp = _place_input(reference, probe, (0, 5, 1, 2, 3, 4), 2, None, rng)
        expected = _paint(np.where(reference, 1, 2).astype(np.int8), reference, matches[1], 3, 4)
        assert np.array_equal(solve(inp), expected)
        mapping = np.arange(9, -1, -1, dtype=np.int8)
        assert np.array_equal(solve(mapping[inp]), mapping[expected])
    bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "reference_classes": len(references), "reference_dimensions": len(dimensions),
            "query_shapes": len(query_shapes), "ranks": sorted(counts), "counter_object_areas": sorted(sizes),
            "palettes": len(palettes), "layouts": len(layouts), "match_counts": sorted(matched)}
