"""Echo chambers: periodic erosion layers restored only in sealed objects.

The solver has no access to generation, seeds, demonstrations, or palette state.
All coordinates and colors used to infer an answer come from its argument.
"""
from collections import deque

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color range")
    return a.astype(np.int8, copy=False)


def _regions(mask, neighbors=N4):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        cells, stack = [], [(r, c)]
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for dr, dc in neighbors:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _holes(body, neighbors=N4):
    """Padding treats the outside of even an edge-touching local box as exterior."""
    a = np.pad(body, 1)
    reached = np.zeros(a.shape, bool)
    reached[0, 0] = True
    queue = deque([(0, 0)])
    h, w = a.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not a[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                queue.append((nr, nc))
    return (~a & ~reached)[1:-1, 1:-1]


def _depth(silhouette):
    """Zero-based orthogonal erosion layers of a hole-filled silhouette."""
    depth = np.full(silhouette.shape, -1, np.int16)
    remaining = silhouette.copy()
    layer = 0
    while remaining.any():
        p = np.pad(remaining, 1)
        core = remaining & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        depth[remaining & ~core] = layer
        remaining = core
        layer += 1
    return depth


def _unambiguous(body):
    a, b, c, d = body[:-1, :-1], body[:-1, 1:], body[1:, :-1], body[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _recover(observed, body, filled, depth):
    """Compare all supported periods; competing full paintings are forbidden.

    A phase needs samples at three different depths, not merely many pixels on
    one shell. The error ceiling is part of the visible family convention.
    """
    colors = np.unique(observed[body])
    _require(len(colors) == 2, "sealed object must use two texture colors")
    limit = max(1, int(body.sum()) // 20)
    candidates = []
    for period in range(2, min(3, (int(depth.max()) + 1) // 3) + 1):
        word, errors = [], 0
        for phase in range(period):
            sample = body & (depth % period == phase)
            if len(np.unique(depth[sample])) < 3:
                break
            values = observed[sample]
            counts = [int(np.count_nonzero(values == color)) for color in colors]
            if counts[0] == counts[1]:
                break
            word.append(int(colors[int(counts[1] > counts[0])]))
            errors += min(counts)
        else:
            if len(set(word)) != 2 or errors > limit:
                continue
            painting = np.zeros(observed.shape, np.int8)
            painting[filled] = np.asarray(word, np.int8)[depth[filled] % period]
            candidates.append((period, tuple(word), errors, painting))
    _require(bool(candidates), "no supported repeating shell pattern")
    chosen = candidates[0]
    _require(all(np.array_equal(chosen[3][filled], x[3][filled]) for x in candidates),
             "ambiguous periodic painting")
    return chosen


def _analyze(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg),
             "background must surround the scene")
    foreground = a != bg
    _require(foreground.any(), "empty scene")
    components = _regions(foreground)
    _require(len(components) == len(_regions(foreground, N8)), "diagonal object contact")
    _require(_unambiguous(foreground), "diagonal leak")
    objects, all_holes = [], np.zeros(a.shape, bool)
    for cells in components:
        rr, cc = np.asarray(cells, int).T
        r0, c0 = int(rr.min()), int(cc.min())
        r1, c1 = int(rr.max()) + 1, int(cc.max()) + 1
        box = (slice(r0, r1), slice(c0, c1))
        body = np.zeros((r1 - r0, c1 - c0), bool)
        body[rr - r0, cc - c0] = True
        local = a[box]
        colors = np.unique(local[body])
        _require(1 <= len(colors) <= 2, "each object has at most two body colors")
        holes = _holes(body)
        _require(np.array_equal(holes, _holes(body, N8)), "ambiguous cavity topology")
        _require(len(_regions(holes)) == len(_regions(holes, N8)), "diagonal cavity contact")
        _require(not np.any((local != bg) & ~body), "interlocking object boxes")
        _require(not all_holes[box][holes].any(), "shared hole ownership")
        all_holes[box] |= holes
        filled = body | holes
        d = _depth(filled)
        o = dict(box=box, body=body, holes=holes, filled=filled, depth=d, colors=tuple(map(int, colors)))
        if holes.any():
            period, word, errors, painting = _recover(local, body, filled, d)
            o.update(period=period, word=word, errors=errors, painting=painting)
        objects.append(o)
    _require(np.array_equal(all_holes, _holes(foreground)), "collective enclosure")
    return a, bg, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, _, objects = _analyze(input_grid)
    answer = a.copy()
    for o in objects:
        if o["holes"].any():
            view = answer[o["box"]]
            view[o["filled"]] = o["painting"][o["filled"]]
    return answer


def _shape(rng, h, w, period, rich=True):
    """A solid silhouette with independent corner steps and exterior bays."""
    for _ in range(40):
        mask = np.ones((h, w), bool)
        if rich:
            for _ in range(int(rng.integers(1, 6))):
                side = int(rng.integers(4))
                depth = int(rng.integers(1, 4))
                if side < 2:
                    start = int(rng.integers(w - 1))
                    end = min(w, start + int(rng.integers(2, 7)))
                    cut = (slice(0, depth) if side == 0 else slice(h - depth, h), slice(start, end))
                else:
                    start = int(rng.integers(h - 1))
                    end = min(h, start + int(rng.integers(2, 7)))
                    cut = (slice(start, end), slice(0, depth) if side == 2 else slice(w - depth, w))
                candidate = mask.copy()
                candidate[cut] = False
                if (len(_regions(candidate)) == 1 and _unambiguous(candidate)
                        and not _holes(candidate).any() and int(_depth(candidate).max()) + 1 >= 3 * period):
                    mask = candidate
        if int(_depth(mask).max()) + 1 >= 3 * period:
            return mask
    raise RuntimeError("silhouette construction exhausted")


def _carve(rng, filled, period, multiple=False, irregular=False, core=False, teaching=False):
    h, w = filled.shape
    d = _depth(filled)
    for _ in range(100):
        holes = np.zeros_like(filled)
        if core:
            # Training hides two complete shells; tests hide at least three.
            threshold = int(d.max()) - 1 if teaching else 3 * period
            if threshold < 3 * period or int(d.max()) - threshold + 1 < (2 if teaching else 3):
                continue
            holes = d >= threshold
            if multiple:
                # A separate L pocket demonstrates independent cavity ownership
                # and irregular interiors in the same body as the missing core.
                hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, 5))
                r, c = int(rng.integers(1, h - hh)), int(rng.integers(1, w - ww))
                pocket = np.ones((hh, ww), bool)
                if irregular:
                    pocket[hh // 2:, ww // 2:] = False
                extra = np.zeros_like(filled)
                extra[r:r + hh, c:c + ww] = pocket
                p = np.pad(holes, 1)
                near = np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w] for dr, dc in N8])
                if np.any(extra & (near | holes | (d < 1))):
                    continue
                holes |= extra
                if len(_regions(holes)) != 2:
                    continue
        else:
            count = 2 if multiple else 1
            for k in range(count):
                hh = int(rng.integers(2, min(6, h // 3) + 1))
                ww = int(rng.integers(2, min(6, w // 3) + 1))
                r = int(rng.integers(1, h - hh))
                c = int(rng.integers(1, w - ww))
                pocket = np.ones((hh, ww), bool)
                if irregular and k == 0:
                    pocket[hh // 2:, ww // 2:] = False
                candidate = np.zeros_like(filled)
                candidate[r:r + hh, c:c + ww] = pocket
                p = np.pad(holes, 1)
                near = np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w] for dr, dc in N8])
                if np.any(candidate & (near | holes | (d < 1))):
                    break
                holes |= candidate
            if len(_regions(holes)) != count:
                continue
        body = filled & ~holes
        if (not holes.any() or len(_regions(body)) != 1 or not _unambiguous(body)
                or not np.array_equal(_holes(body), holes)
                or not np.array_equal(_holes(body, N8), holes)):
            continue
        if not core and not np.any(body & (d == d.max())):
            continue
        if all(len(np.unique(d[body & (d % period == phase)])) >= 3 for phase in range(period)):
            return body, holes
    raise RuntimeError("cavity construction exhausted")


def _open(body):
    """Open the pocket by a straight orthogonal door; preserve one connected body."""
    holes = _holes(body)
    choices = []
    h, w = body.shape
    for r, c in np.argwhere(holes):
        r, c = int(r), int(c)
        for cut in ((slice(0, r + 1), c), (slice(r, h), c),
                    (r, slice(0, c + 1)), (r, slice(c, w))):
            choices.append((int(body[cut].sum()), cut))
    for _, cut in sorted(choices, key=lambda item: item[0]):
        candidate = body.copy()
        candidate[cut] = False
        if (len(_regions(candidate)) == 1 and _unambiguous(candidate)
                and not _holes(candidate).any() and not _holes(candidate, N8).any()):
            return candidate
    raise AssertionError("no unambiguous door")


def _texture(rng, body, holes, filled, period, colors, common=None):
    d = _depth(filled)
    # All nonconstant binary words of the chosen (prime) period are primitive.
    code = int(rng.integers(1, 2 ** period - 1))
    word = np.asarray([colors[(code >> phase) & 1] for phase in range(period)], np.int8)
    clean = np.zeros(body.shape, np.int8)
    clean[filled] = word[d[filled] % period]
    if len(np.unique(clean[holes])) < 2:
        raise RuntimeError("cavity needs at least two texture phases")
    eligible = body.copy() if common is None else body & common
    # Preserve strict majority on every individual observed shell as well.
    shell_sizes = np.bincount(d[body], minlength=int(d.max()) + 1)
    eligible &= shell_sizes[d.clip(min=0)] >= 6
    cells = np.argwhere(eligible)
    max_errors = min(4, max(1, int(body.sum()) // 80), len(cells))
    n = int(rng.integers(1, max_errors + 1))
    selected = rng.choice(len(cells), size=n, replace=False)
    noisy = clean.copy()
    for r, c in cells[selected]:
        noisy[r, c] = colors[1] if clean[r, c] == colors[0] else colors[0]
    recovered = _recover(noisy, body, filled, d)
    if recovered[0] != period or not np.array_equal(recovered[3][filled], clean[filled]):
        raise RuntimeError("competing periodic explanation")
    return noisy, clean


def _chamber(rng, bg, period=2, compact=False, multiple=False, irregular=False,
             core=False, contrast=False, teaching=False):
    for _ in range(80):
        if core:
            side_min = 6 * period + (4 if teaching else 6)
            h = int(rng.integers(max(22, side_min), 29))
            w = int(rng.integers(max(22, side_min), 29))
        elif compact:
            h, w = int(rng.integers(12, 14)), int(rng.integers(12, 14))
        else:
            low = 6 * period + 2
            h, w = int(rng.integers(low, low + 4)), int(rng.integers(low, low + 4))
        filled = _shape(rng, h, w, period, rich=not compact or rng.random() < .5)
        if teaching and filled.all():
            continue
        try:
            body, holes = _carve(rng, filled, period, multiple, irregular, core, teaching)
            opened = _open(body) if contrast else None
            colors = tuple(map(int, rng.choice([c for c in range(10) if c != bg], 2, replace=False)))
            noisy, _ = _texture(rng, body, holes, filled, period, colors, opened)
        except (RuntimeError, AssertionError):
            continue
        image = np.where(body, noisy, bg).astype(np.int8)
        images = [image]
        if contrast:
            images.append(np.where(opened, noisy, bg).astype(np.int8))
        rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
        return [(np.fliplr(np.rot90(a, rotation)) if reflection else np.rot90(a, rotation)).copy()
                for a in images]
    raise RuntimeError("chamber construction exhausted")


def _solid(rng, bg, small=True):
    h, w = (int(rng.integers(3, 7)), int(rng.integers(3, 7))) if small else (8, 8)
    body = np.ones((h, w), bool)
    # Deep exterior bay, with separated runs but no sealed cavity.
    left, right = 1, w - 1
    body[:h - 1, left:right] = False
    body = np.rot90(body, int(rng.integers(4)))
    colors = rng.choice([c for c in range(10) if c != bg], 2, replace=False)
    image = np.where(rng.integers(2, size=body.shape), colors[0], colors[1])
    return np.where(body, image, bg).astype(np.int8)


def _pack(rng, images, bg):
    """Free bounding-box placement, finite randomized search and shelf fallback."""
    order = sorted(range(len(images)), key=lambda i: -images[i].size)
    min_h = max(a.shape[0] for a in images) + 2
    min_w = max(a.shape[1] for a in images) + 2
    area = sum(a.size for a in images)
    for attempt in range(25):
        if attempt < 12:
            h = int(rng.integers(min_h, 31))
            w = int(rng.integers(min_w, 31))
            if (h - 2) * (w - 2) < area:
                continue
        else:
            h = w = 30
        occupied = np.zeros((h, w), bool)
        a = np.full((h, w), bg, np.int8)
        for i in order:
            image = images[i]
            mh, mw = image.shape
            offsets = []
            for r in range(1, h - mh):
                for c in range(1, w - mw):
                    if not occupied[r:r + mh, c:c + mw].any():
                        offsets.append((r, c))
            if not offsets:
                break
            r, c = offsets[int(rng.integers(len(offsets)))]
            a[r:r + mh, c:c + mw] = image
            occupied[max(0, r - 1):min(h, r + mh + 1), max(0, c - 1):min(w, c + mw + 1)] = True
        else:
            return a
    # This covers compact 2x2 arrays and the matched pair, with no hidden layout semantics.
    a = np.full((30, 30), bg, np.int8)
    r, c, shelf_h = 1, 1, 0
    for i in order:
        image = images[i]
        mh, mw = image.shape
        if c + mw > 29:
            r, c, shelf_h = r + shelf_h + 1, 1, 0
        if r + mh > 29:
            raise RuntimeError("objects do not fit")
        a[r:r + mh, c:c + mw] = image
        c += mw + 1
        shelf_h = max(shelf_h, mh)
    return a


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    # Palette meanings are geometric; backgrounds and body colors can vary per example.
    backgrounds = rng.permutation(10)
    use_three = bool(rng.random() < .7)
    shift = str(rng.choice(("hidden_core", "count")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            bg = int(backgrounds[(index + (nt if split == "test" else 0)) % 10])
            for attempt in range(35):
                try:
                    if split == "train" and index == 0:
                        images = _chamber(rng, bg, compact=True, contrast=True)
                    elif split == "train" and index == 1:
                        p = 3 if use_three else 2
                        images = _chamber(rng, bg, p, multiple=True, irregular=True, core=True, teaching=True)
                    elif split == "test" and shift == "hidden_core":
                        p = int(rng.choice((2, 3))) if use_three else 2
                        images = _chamber(rng, bg, p, core=True)
                    elif split == "test" and shift == "count":
                        images = []
                        for _ in range(int(rng.integers(2, 5))):
                            images.extend(_chamber(rng, bg, compact=True, irregular=True))
                    else:
                        p = int(rng.choice((2, 3))) if use_three else 2
                        images = _chamber(rng, bg, p, multiple=rng.random() < .5, irregular=True)
                        if p == 2:
                            for _ in range(int(rng.integers(0, 3))):
                                images.append(_solid(rng, bg))
                    a = _pack(rng, images, bg)
                    # Independent local rotations and reflections occur before packing.
                    # The full scene can also transpose, preserving the isotropic rule.
                    if rng.random() < .5:
                        a = a.T.copy()
                    key = (a.shape, a.tobytes())
                    if key in seen:
                        continue
                    output = solve(a)
                    break
                except (RuntimeError, AssertionError):
                    continue
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(key)
            puzzle[split].append({"input": a, "output": output})
    return puzzle


def _features(a):
    _, bg, objects = _analyze(a)
    hollows = [o for o in objects if o["holes"].any()]
    return {
        "shape": tuple(a.shape), "background": bg, "objects": len(objects), "hollows": len(hollows),
        "periods": tuple(o["period"] for o in hollows),
        "holes": tuple(len(_regions(o["holes"])) for o in hollows),
        "depths": tuple(int(o["depth"].max()) + 1 for o in hollows),
        "irregular_holes": sum(len(cells) != (max(r for r, c in cells) - min(r for r, c in cells) + 1)
                               * (max(c for r, c in cells) - min(c for r, c in cells) + 1)
                               for o in hollows for cells in _regions(o["holes"])),
        "open_bays": sum(not o["holes"].any() and any(np.any(np.diff(np.flatnonzero(row)) > 1)
                         for rows in (o["body"], o["body"].T) for row in rows) for o in objects),
        "hidden_core": sum(not np.any(o["body"] & (o["depth"] == o["depth"].max())) for o in hollows),
        "missing_depths": tuple(int(o["depth"].max()) + 1 - len(np.unique(o["depth"][o["body"]]))
                                for o in hollows),
        "steps": sum(not o["filled"].all() for o in hollows),
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "dimensions changed")
            key = (a.shape, a.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _, bg, objects = _analyze(a)
            changed_zone = np.zeros(a.shape, bool)
            for o in objects:
                inp, result = a[o["box"]], out[o["box"]]
                if not o["holes"].any():
                    _require(np.array_equal(inp[o["body"]], result[o["body"]]), "cavity-free body changed")
                    continue
                changed_zone[o["box"]] |= o["filled"]
                _require(np.array_equal(result[o["filled"]], o["painting"][o["filled"]]), "shell painting")
                _require(np.all(np.isin(result[o["holes"]], o["colors"])), "incomplete cavity fill")
                flips = inp[o["body"]] != result[o["body"]]
                _require(int(flips.sum()) == o["errors"] and o["errors"] >= 1, "missing sparse repair")
                _require(len(np.unique(result[o["holes"]])) == 2, "cavity must reveal the periodic texture")
            _require(np.array_equal(a[~changed_zone], out[~changed_zone]), "exterior changed")
            _require(not np.array_equal(a, out), "identity pair")
            features[split].append(_features(a))
    train = features["train"]
    _require(any(f["open_bays"] for f in train), "open-bay contrast missing")
    _require(any(f["irregular_holes"] for f in train) and any(max(f["holes"], default=0) >= 2 for f in train),
             "irregular/multiple cavities must be demonstrated")
    _require(any(f["hidden_core"] for f in train), "missing complete layers must be demonstrated")
    _require(any(f["steps"] for f in train), "nonrectangular layer coordinates must be demonstrated")
    trained_periods = {p for f in train for p in f["periods"]}
    _require(all(set(f["periods"]) <= trained_periods for f in features["test"]), "unseen test period")
    _require(len({f["objects"] for f in train + features["test"]}) >= 2, "constant object count")
    max_train_closed = max(f["hollows"] for f in train)
    max_train_missing = max(max(f["missing_depths"], default=0) for f in train)
    _require(all(f["hollows"] > max_train_closed or max(f["missing_depths"], default=0) > max_train_missing
                 for f in features["test"]), "no structural train/test shift")
    return None


def self_test():
    """Bounded pixel checks, ablations, invalid counts, and 42-seed diversity."""
    # Hand-derived alternating concentric squares. The 4x4 hole removes all
    # observations of depths 6 and 7; its center must still get the odd color.
    rr, cc = np.indices((16, 16))
    square_depth = np.minimum.reduce((rr, cc, 15 - rr, 15 - cc))
    clean = np.where(square_depth % 2 == 0, 2, 7).astype(np.int8)
    damaged = clean.copy()
    damaged[6:10, 6:10] = 0
    damaged[0, 5] = 7
    sealed = np.pad(damaged, 1)
    _require(np.array_equal(solve(sealed), np.pad(clean, 1)), "manual missing-core regression")
    opened = damaged.copy()
    opened[:7, 7] = 0
    opened = np.pad(opened, 1)
    _require(np.array_equal(solve(opened), opened), "manual open-door preservation")
    # Transposition and a bijective recoloring cannot change geometric roles.
    recolor = np.asarray([8, 0, 4, 1, 2, 3, 5, 9, 6, 7], np.int8)
    _require(np.array_equal(solve(recolor[sealed.T]), recolor[np.pad(clean, 1).T]),
             "manual color/axis generalization")
    shapes, layouts, silhouettes, palettes, periods, object_counts, signatures = (set() for _ in range(7))
    totals = dict(irregular_holes=0, open_bays=0, hidden_core=0, steps=0, multiple=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = []
        for split in puzzle:
            for pair in puzzle[split]:
                a = pair["input"]
                _require(np.array_equal(solve(a.tolist()), pair["output"]), "input-only solve")
                _, bg, objects = _analyze(a)
                f = _features(a)
                shapes.add(a.shape)
                layouts.add((a.shape, (a != bg).tobytes()))
                object_counts.add((f["objects"], f["hollows"]))
                periods.update(f["periods"])
                palettes.update(tuple(o["colors"]) for o in objects)
                for o in objects:
                    silhouettes.add((o["body"].shape, o["body"].tobytes()))
                for name in totals:
                    totals[name] += int(max(f["holes"], default=0) > 1) if name == "multiple" else f[name]
                signature.append((a.shape, a.tobytes(), pair["output"].tobytes()))
        signatures.add(tuple(signature))
        # A omitted: leave original bodies and uniformly fill cavities. Every
        # closed body contains genuine corrections and a nonuniform cavity.
        for pair in puzzle["train"]:
            a, bg, objects = _analyze(pair["input"])
            ablated = a.copy()
            for o in objects:
                if o["holes"].any():
                    ablated[o["box"]][o["holes"]] = o["colors"][0]
            _require(not np.array_equal(ablated, pair["output"]), "periodicity dispensable")
        # B omitted: restore the open copy with the same pattern as the sealed
        # partner. It retains genuine damage, so even repainting only its body
        # produces a wrong answer; filling its bay would also be wrong.
        first = puzzle["train"][0]
        a, bg, objects = _analyze(first["input"])
        closed = next(o for o in objects if o["holes"].any())
        opened = next(o for o in objects if not o["holes"].any())
        _require(closed["body"].shape == opened["body"].shape, "contrast geometry")
        _require(np.any(closed["painting"][opened["body"]] != a[opened["box"]][opened["body"]]),
                 "topology dispensable")
        # Cavity-induced boundaries must disagree with the intended outer depth.
        _require(any(np.any(np.asarray(o["word"])[_depth(o["body"])[o["body"]] % o["period"]]
                           != o["painting"][o["body"]])
                     for o in objects if o["holes"].any()), "completion does not affect metric")
        # Copying the deepest visible layer's color into the missing core is
        # insufficient even when all other pixels are predicted correctly.
        second = puzzle["train"][1]
        a, _, objects = _analyze(second["input"])
        o = next(o for o in objects if o["holes"].any())
        last = int(o["depth"][o["body"]].max())
        missing = o["holes"] & (o["depth"] > last)
        _require(missing.any() and np.any(o["painting"][missing] != o["word"][last % o["period"]]),
                 "nearest-layer copying not distinguished")
    _require(len(signatures) == 42 and len(shapes) >= 20 and len(layouts) >= 180, "scene diversity")
    _require(len(silhouettes) >= 150 and len(palettes) >= 30 and len(object_counts) >= 6, "object diversity")
    _require(periods == {2, 3} and all(v >= 10 for v in totals.values()), "concept coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2468 + nt * 7 + nq, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    x, y = generate_puzzle(987, 2, 1), generate_puzzle(987, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in x for p, q in zip(x[s], y[s])
                 for k in ("input", "output")), "nondeterminism")
    # Corrupt a preserved open body, a repaired shell, and a hole separately.
    base = generate_puzzle(0, 2, 1)
    for role in ("open", "body", "hole", "background"):
        altered = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in base[s]] for s in base}
        pair = altered["train"][0]
        _, bg, objects = _analyze(pair["input"])
        if role == "background":
            pair["output"][0, 0] = (bg + 1) % 10
        else:
            o = next(o for o in objects if bool(o["holes"].any()) != (role == "open"))
            mask = o["holes"] if role == "hole" else o["body"]
            r, c = np.argwhere(mask)[0]
            view = pair["output"][o["box"]]
            view[r, c] = (int(view[r, c]) + 1) % 10
        try:
            validate_generated_puzzle(altered)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted wrong " + role)
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "layouts": len(layouts),
            "silhouettes": len(silhouettes), "palettes": len(palettes),
            "object_count_combinations": len(object_counts), "periods": sorted(periods), **totals}
