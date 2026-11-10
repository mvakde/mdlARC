"""Cavity stencils: negative-space motifs texture their own enclosing bodies.

No parent imports, puzzle-level hidden conventions, or generation state are
used by solve. Colors 2/6/8 and the local [1,2,0] tile order are fixed rules.
"""
from itertools import permutations

import numpy as np


OUTLINE, ACCENT, CARRIER = 2, 6, 8
INPUT_COLORS = (0, 1, 3, 4, 5, 7, 9)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _regions(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, queue = {start}, [start]
        for r, c in queue:
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    queue.append(q)
        result.append(cells)
    return result


def _holes(mask, neighbors=N4):
    # An outer solid sentinel ring permits inexpensive flat-index flood fill.
    # Inside it, the extra empty ring represents the object's true exterior.
    padded = np.pad(mask, 2, constant_values=True)
    padded[1:-1, 1:-1] = np.pad(mask, 1)
    h, w = padded.shape
    blocked = padded.ravel().tolist()
    exterior = [False] * (h * w)
    start = w + 1
    exterior[start] = True
    queue = [start]
    offsets = tuple(dr * w + dc for dr, dc in neighbors)
    for cell in queue:
        for offset in offsets:
            other = cell + offset
            if not blocked[other] and not exterior[other]:
                exterior[other] = True
                queue.append(other)
    outside = np.array(exterior, dtype=bool).reshape(h, w)
    return ~mask & ~outside[2:-2, 2:-2]


def _crop(mask):
    rr, cc = np.nonzero(mask)
    _require(len(rr) > 0, "empty mask")
    r, c = int(rr.min()), int(cc.min())
    return mask[r:int(rr.max()) + 1, c:int(cc.max()) + 1].copy(), (r, c)


def _unambiguous(mask):
    p = np.pad(mask, 1)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _neighbors(mask, diagonal=True):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                                 for dr, dc in (N8 if diagonal else N4)])


def _phase_coverage(mask):
    rr, cc = np.nonzero(mask)
    return len(set(zip((rr % 3).tolist(), (cc % 3).tolist())))


def _separated_runs(mask):
    return any(np.any(np.diff(np.flatnonzero(row)) > 1) for axis in (mask, mask.T)
               for row in axis if row.sum() > 1)


def _stencil(mask):
    holes = _holes(mask)
    return _crop(holes)[0] if holes.any() else None


def _orient(rng, mask):
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return _crop(mask)[0]


# Exhaust the legal negative-space stencil space rather than a motif vocabulary.
_mask_list = []
for _code in range(1, 511):
    _m = np.array([(_code >> k) & 1 for k in range(9)], bool).reshape(3, 3)
    if not all((_m[0].any(), _m[2].any(), _m[:, 0].any(), _m[:, 2].any())):
        continue
    _body = np.ones((5, 5), bool)
    _body[1:4, 1:4] = ~_m
    if _unambiguous(_m) and len(_regions(_body)) == 1:
        _mask_list.append(_m)
MASKS = np.array(_mask_list)
DENSITIES = MASKS.sum(axis=(1, 2))
HOLE_COUNTS = np.array([len(_regions(m)) for m in MASKS])
MASK_INDEX = {m.tobytes(): i for i, m in enumerate(MASKS)}
ORDERS = tuple((t, rows, cols) for t in (False, True)
               for rows in permutations(range(3)) for cols in permutations(range(3)))
DIRECT = ORDERS.index((False, (1, 2, 0), (1, 2, 0)))


def _ordered(mask, order):
    transpose, rows, cols = ORDERS[order]
    return (mask.T if transpose else mask)[np.ix_(rows, cols)]


AGREEMENT = np.array([[np.array_equal(_ordered(m, k), _ordered(m, DIRECT))
                       for k in range(len(ORDERS))] for m in MASKS])


def _second_candidates(first):
    return ((HOLE_COUNTS >= 2) & (DENSITIES != DENSITIES[first])
            & ((AGREEMENT & AGREEMENT[first]).sum(axis=1) == 1))


FIRST_IDS = np.array([i for i in range(len(MASKS)) if HOLE_COUNTS[i] == 1
                      and 5 <= DENSITIES[i] <= 7 and _second_candidates(i).any()
                      and not np.array_equal(MASKS[i], MASKS[i][::-1])
                      and not np.array_equal(MASKS[i], MASKS[i][:, ::-1])])


def _choose_tile(rng, allowed=None):
    ids = np.arange(len(MASKS)) if allowed is None else np.flatnonzero(allowed)
    _require(len(ids) > 0, "no eligible stencil")
    density = int(rng.choice(np.unique(DENSITIES[ids])))
    return MASKS[int(rng.choice(ids[DENSITIES[ids] == density]))].copy()


def _closed(rng, tile, size="ordinary", witness=False):
    upper = 2 if size == "small" else 3
    top, bottom, left, right = map(int, rng.integers(1, upper + 1, size=4))
    if witness:
        walls = [top, bottom, left, right]
        walls[int(rng.integers(4))] = 3
        top, bottom, left, right = walls
    if size == "long":
        if rng.integers(2):
            bottom = int(rng.integers(17, 21)) - 3 - top
        else:
            right = int(rng.integers(17, 21)) - 3 - left
    mask = np.ones((top + 3 + bottom, left + 3 + right), bool)
    mask[top:top + 3, left:left + 3] = ~tile
    original_holes = _holes(mask)
    # Safe silhouette edits preserve every hole and ownership. Neither the
    # source patch nor the resulting texture is a rectangular-frame template.
    for _ in range(int(rng.integers(1, 6))):
        candidate = mask.copy()
        h, w = mask.shape
        side = int(rng.integers(4))
        if side < 2:
            start = int(rng.integers(w))
            end = min(w, start + int(rng.integers(1, 4)))
            cut = (slice(0, 1) if side == 0 else slice(h - 1, h), slice(start, end))
        else:
            start = int(rng.integers(h))
            end = min(h, start + int(rng.integers(1, 4)))
            cut = (slice(start, end), slice(0, 1) if side == 2 else slice(w - 1, w))
        candidate[cut] = False
        if (candidate.any() and _unambiguous(candidate) and len(_regions(candidate)) == 1
                and np.array_equal(_holes(candidate), original_holes)
                and (not witness or _phase_coverage(_crop(candidate)[0]) == 9)):
            mask = candidate
    expanded = np.pad(mask, 1)
    reference = np.pad(original_holes, 1)
    h, w = mask.shape
    for _ in range(int(rng.integers(1, 5))):
        candidate = expanded.copy()
        side = int(rng.integers(4))
        axis = w if side < 2 else h
        start = int(rng.integers(1, axis + 1))
        end = min(axis + 1, start + int(rng.integers(1, 4)))
        if side < 2:
            candidate[(slice(0, 2) if side == 0 else slice(h, h + 2)), start:end] = True
        else:
            candidate[start:end, (slice(0, 2) if side == 2 else slice(w, w + 2))] = True
        cropped = _crop(candidate)[0]
        if (_unambiguous(candidate) and len(_regions(candidate)) == 1
                and np.array_equal(_holes(candidate), reference)
                and (not witness or _phase_coverage(cropped) == 9)):
            expanded = candidate
    return _crop(expanded)[0]


def _open_counterpart(rng, closed):
    _require(len(_regions(_holes(closed))) == 1, "contrast needs one cavity")
    for rotation in rng.permutation(4):
        candidate = np.rot90(closed, int(rotation)).copy()
        holes = _holes(candidate)
        r = int(np.nonzero(holes)[0].min())
        for c in rng.permutation(np.flatnonzero(holes[r])):
            opened = candidate.copy()
            opened[:r + 1, int(c)] = False
            if (_unambiguous(opened) and len(_regions(opened)) == 1
                    and not _holes(opened).any() and _separated_runs(opened)):
                return _crop(np.rot90(opened, -int(rotation)))[0]
    # Teaching bodies always retain an orthogonal escape candidate, but an
    # explicit failure allows the bounded caller to resample its geometry.
    return None


def _solid(rng, small=False):
    maximum = 6 if small else 9
    h, w = map(int, rng.integers(1, maximum + 1, size=2))
    style = int(rng.integers(5))
    mask = np.ones((h, w), bool)
    if style == 0:
        mask = np.ones((1, int(rng.integers(1, maximum + 1))), bool)
    elif style == 1 and min(h, w) >= 3:
        l = int(rng.integers(1, w - 1))
        r = int(rng.integers(l + 1, w))
        mask[:int(rng.integers(1, h)), l:r] = False
    elif style == 2:
        mask[:] = False
        mask[:int(rng.integers(1, h + 1)), :] = True
        mask[:, :int(rng.integers(1, w + 1))] = True
    elif style == 3:
        mask[:] = False
        mask[h // 2, w // 2] = True
        for _ in range(int(rng.integers(4, 30))):
            r, c = np.argwhere(mask)[int(rng.integers(mask.sum()))]
            dr, dc = N4[int(rng.integers(4))]
            y, x = int(r + dr), int(c + dc)
            if 0 <= y < h and 0 <= x < w:
                candidate = mask.copy()
                candidate[y, x] = True
                if _unambiguous(candidate) and not _holes(candidate).any():
                    mask = candidate
    return _orient(rng, mask)


def _pack(rng, masks, shape, margin, gap, attempts=5):
    h, w = shape
    for _ in range(attempts):
        blocked = np.zeros(shape, bool)
        scene = np.zeros(shape, bool)
        order = sorted(range(len(masks)), key=lambda i: -masks[i].size)
        for i in order:
            mask = masks[i]
            mh, mw = mask.shape
            if mh + 2 * margin > h or mw + 2 * margin > w:
                break
            prefix = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = prefix[mh:, mw:] - prefix[:-mh, mw:] - prefix[mh:, :-mw] + prefix[:-mh, :-mw]
            valid = sums == 0
            if margin:
                valid[:margin] = valid[-margin:] = False
                valid[:, :margin] = valid[:, -margin:] = False
            positions = np.argwhere(valid)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            scene[r:r + mh, c:c + mw] |= mask
            blocked[max(0, r - gap):min(h, r + mh + gap),
                    max(0, c - gap):min(w, c + mw + gap)] = True
        else:
            if scene.mean() < .48:
                return scene
    return None


def _scene(rng, required, optional):
    masks = required + optional
    area = sum(m.size for m in masks)
    ratio = float(np.exp(rng.uniform(-.6, .6)))
    target = area / float(rng.uniform(.22, .4))
    margin, gap = int(rng.choice([0, 0, 1, 2, 3])), int(rng.choice([1, 1, 2, 3]))
    h = max(6, int(round(np.sqrt(target * ratio))), max(m.shape[0] for m in masks) + 2 * margin)
    w = max(6, int(round(np.sqrt(target / ratio))), max(m.shape[1] for m in masks) + 2 * margin)
    for extra in (0, 4, 10, 30):
        scene = _pack(rng, masks, (min(h + extra, 30), min(w + extra, 30)), margin, gap)
        if scene is not None:
            return scene
    # Capacity fallback preserves stencil identities and all required anchors.
    smaller = []
    for m in optional:
        tile = _stencil(m)
        if tile is None:
            smaller.append(np.ones((1, 1), bool))
        else:
            body = np.ones((5, 5), bool)
            body[1:4, 1:4] = ~tile
            smaller.append(body)
    masks = required + smaller
    scene = _pack(rng, masks, (30, 30), 0, 1, attempts=15)
    if scene is not None:
        return scene
    scene = np.zeros((30, 30), bool)
    row, col, shelf_height = 0, 0, 0
    for mask in masks:
        mh, mw = mask.shape
        if col + mw > 30:
            row, col, shelf_height = row + shelf_height + 1, 0, 0
        _require(row + mh <= 30, "shelf capacity exceeded")
        scene[row:row + mh, col:col + mw] = mask
        col += mw + 1
        shelf_height = max(shelf_height, mh)
    _require(scene.mean() < .5, "fallback has no background majority")
    return scene


def _analyze(input_grid):
    grid = np.asarray(input_grid)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)), "invalid cells")
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 2 and counts.max() > grid.size / 2, "expected binary strict-majority scene")
    bg, fg = int(colors[np.argmax(counts)]), int(colors[np.argmin(counts)])
    _require(bg in INPUT_COLORS and fg in INPUT_COLORS, "reserved input color")
    body = grid == fg
    _require(_unambiguous(body), "diagonal-only topology")
    components = _regions(body)
    _require(len(components) == len(_regions(body, N8)), "diagonal object contact")
    all_holes = np.zeros_like(body)
    objects = []
    for cells in components:
        rr, cc = np.array(sorted(cells)).T
        r0, c0 = int(rr.min()), int(cc.min())
        local = np.zeros((int(rr.max()) - r0 + 1, int(cc.max()) - c0 + 1), bool)
        local[rr - r0, cc - c0] = True
        holes = _holes(local)
        _require(np.array_equal(holes, _holes(local, N8)), "ambiguous cavities")
        tile = None
        if holes.any():
            tile, _ = _crop(holes)
            _require(tile.shape == (3, 3) and tile.tobytes() in MASK_INDEX, "invalid cavity stencil")
            hr, hc = np.nonzero(holes)
            _require(not body[hr + r0, hc + c0].any(), "nested object")
            _require(not all_holes[hr + r0, hc + c0].any(), "shared cavity")
            all_holes[hr + r0, hc + c0] = True
        objects.append({"mask": local, "holes": holes, "tile": tile, "origin": (r0, c0),
                        "rows": rr, "cols": cc, "hole_count": len(_regions(holes))})
    _require(np.array_equal(all_holes, _holes(body)), "collective enclosure")
    exterior = ~body & ~all_holes
    return {"grid": grid, "background": bg, "foreground": fg, "body": body,
            "holes": all_holes, "objects": objects, "exterior": exterior,
            "outline": _neighbors(body) & exterior}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    result = info["grid"].astype(np.int8, copy=True)
    for obj in info["objects"]:
        if obj["tile"] is not None:
            r0, c0 = obj["origin"]
            rr, cc = obj["rows"], obj["cols"]
            pattern = obj["tile"][(rr - r0 + 1) % 3, (cc - c0 + 1) % 3]
            result[rr, cc] = np.where(pattern, ACCENT, CARRIER)
    result[info["holes"]] = ACCENT
    result[info["outline"]] = OUTLINE
    return result


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be legal integers")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["count", "scale"]))
    variable_palette = bool(rng.random() < .75)
    fixed_palette = tuple(map(int, rng.choice(INPUT_COLORS, 2, replace=False)))
    first = int(rng.choice(FIRST_IDS))
    # Orienting the body also orients its negative stencil. Select the second
    # witness against the actual first stencil, not a pre-rotation flag.
    closed = None
    for _ in range(24):
        candidate = _orient(rng, _closed(rng, MASKS[first], witness=True))
        opened = _open_counterpart(rng, candidate)
        actual = MASK_INDEX[_stencil(candidate).tobytes()]
        if opened is not None and _second_candidates(actual).any():
            closed = candidate
            break
    if closed is None:
        closed = np.ones((9, 9), bool)
        closed[3:6, 3:6] = ~MASKS[first]
        opened = _open_counterpart(rng, closed)
        _require(opened is not None, "no open contrast")
        actual = first
    second_tile = _choose_tile(rng, _second_candidates(actual))
    second = _closed(rng, second_tile, witness=True)
    alternatives = []
    for turn in range(4):
        for flip in (False, True):
            variant = np.rot90(second, turn)
            if flip:
                variant = np.fliplr(variant)
            index = MASK_INDEX[_stencil(variant).tobytes()]
            if _second_candidates(actual)[index]:
                alternatives.append(variant.copy())
    second = alternatives[int(rng.integers(len(alternatives)))]
    count0 = int(rng.integers(2, 5))
    count1 = int(rng.choice([n for n in range(2, 5) if n != count0]))
    puzzle, used, train_tiles = {"train": [], "test": []}, set(), set()
    first_palette = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            required = []
            if split == "train" and index == 0:
                required, n = [closed, opened], count0
            elif split == "train" and index == 1:
                required, n = [second, _solid(rng)], count1
            elif split == "train":
                n = int(rng.integers(2, 5))
            else:
                n = int(rng.integers(5, 7)) if regime == "count" else int(rng.integers(1, 5))
                allowed = np.array([m.tobytes() not in train_tiles for m in MASKS])
                tile = _choose_tile(rng, allowed)
                required = [_orient(rng, _closed(rng, tile, "small" if regime == "count" else "long"))]
            optional = []
            for _ in range(n - len(required)):
                if rng.random() < .55:
                    allowed = (np.array([m.tobytes() not in train_tiles for m in MASKS])
                               if split == "test" else None)
                    tile = _choose_tile(rng, allowed)
                    # Random rotations could accidentally turn an unseen tile
                    # into a seen one; filter the visible result below.
                    mask = _closed(rng, tile, "small" if split == "test" and regime == "count" else "ordinary")
                    orientations = [np.rot90(mask, k) for k in range(4)]
                    orientations += [np.fliplr(m) for m in orientations]
                    eligible = [m for m in orientations if split == "train" or _stencil(m).tobytes() not in train_tiles]
                    optional.append(eligible[int(rng.integers(len(eligible)))].copy())
                else:
                    optional.append(_solid(rng, small=split == "test" and regime == "count"))
            # The required test tile obeys the same visible novelty constraint.
            if split == "test" and _stencil(required[0]).tobytes() in train_tiles:
                tile = _choose_tile(rng, np.array([m.tobytes() not in train_tiles for m in MASKS]))
                required[0] = _closed(rng, tile, "small" if regime == "count" else "long")
            if variable_palette:
                bg, fg = map(int, rng.choice(INPUT_COLORS, 2, replace=False))
                if split == "train" and index == 1:
                    available = [c for c in INPUT_COLORS if c not in first_palette]
                    bg, fg = map(int, rng.choice(available, 2, replace=False))
            else:
                bg, fg = fixed_palette
            if split == "train" and index == 0:
                first_palette = (bg, fg)
            for _ in range(24):
                scene = _scene(rng, required, optional)
                inp = np.where(scene, fg, bg).astype(np.int8)
                if _key(inp) not in used:
                    break
            else:
                # At most thirteen previous inputs, but 42 visible palettes.
                # This fallback changes observable data, never the solve rule.
                for bg2 in INPUT_COLORS:
                    for fg2 in INPUT_COLORS:
                        if bg2 == fg2:
                            continue
                        candidate = np.where(scene, fg2, bg2).astype(np.int8)
                        if _key(candidate) not in used:
                            inp = candidate
                            break
                    else:
                        continue
                    break
            used.add(_key(inp))
            puzzle[split].append({"input": inp, "output": solve(inp)})
            if split == "train":
                train_tiles.update(o["tile"].tobytes() for o in _analyze(inp)["objects"] if o["tile"] is not None)
    return puzzle


def _observed_orders(infos, outputs):
    candidates = set(range(len(ORDERS)))
    for info, output in zip(infos, outputs):
        for obj in info["objects"]:
            if obj["tile"] is None:
                continue
            r0, c0 = obj["origin"]
            rr, cc = obj["rows"], obj["cols"]
            observed = output[rr, cc] == ACCENT
            candidates &= {k for k in candidates if np.array_equal(
                _ordered(obj["tile"], k)[(rr - r0) % 3, (cc - c0) % 3], observed)}
    return candidates


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, inspections, outputs = set(), {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            _require(out.shape == inp.shape and out.ndim == 2, "output dimensions changed")
            _require(np.issubdtype(out.dtype, np.integer) and np.all((0 <= out) & (out <= 9)), "invalid output cells")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            _require(1 <= len(info["objects"]) <= 6, "unsupported generated object count")
            expected = inp.copy()
            for obj in info["objects"]:
                r0, c0 = obj["origin"]
                rr, cc = obj["rows"], obj["cols"]
                if obj["tile"] is None:
                    _require(np.all(out[rr, cc] == info["foreground"]), "cavity-free material changed")
                else:
                    # Validate by normalized tile lookup on cropped body axes.
                    tile = _ordered(obj["tile"], DIRECT)
                    local_rows, local_cols = np.indices(obj["mask"].shape)
                    texture = np.where(tile[local_rows % 3, local_cols % 3], ACCENT, CARRIER)
                    view = expected[r0:r0 + len(texture), c0:c0 + texture.shape[1]]
                    view[obj["mask"]] = texture[obj["mask"]]
                    _require(np.all(out[rr, cc] == texture[obj["mask"]]), "incorrect local body texture")
            expected[info["holes"]] = ACCENT
            expected[info["outline"]] = OUTLINE
            _require(np.all(out[info["holes"]] == ACCENT), "incomplete original cavity fill")
            _require(np.all(out[info["outline"]] == OUTLINE), "incorrect exterior outline")
            _require(np.all(out[info["exterior"] & ~info["outline"]] == info["background"]), "distant background changed")
            _require(np.array_equal(expected, out), "incorrect transformation")
            inspections[split].append(info)
            outputs[split].append(out)
    train = inspections["train"]
    _require(_observed_orders(train, outputs["train"]) == {DIRECT}, "training phase/order ambiguity")
    _require(any(o["tile"] is None and _separated_runs(o["mask"]) for i in train for o in i["objects"]), "no open-bay witness")
    _require(any(o["hole_count"] >= 2 for i in train for o in i["objects"]), "no multi-cavity witness")
    _require(any(o["hole_count"] == 1 for i in train for o in i["objects"]), "no single-cavity witness")
    _require(any(np.any(o["mask"][:-1, :-1] & o["mask"][1:, :-1]
                           & o["mask"][:-1, 1:] & o["mask"][1:, 1:])
                 for i in train for o in i["objects"] if o["tile"] is not None), "no thick-body witness")
    _require(any(np.any(i["outline"] & ~_neighbors(i["body"], False)) for i in train), "eight-neighbor outline not distinguished")
    training_tiles = {o["tile"].tobytes() for i in train for o in i["objects"] if o["tile"] is not None}
    _require(len(training_tiles) >= 2, "insufficient motif variation")
    _require(len({len(i["objects"]) for i in train}) >= 2, "training counts do not vary")
    maximum_count = max(len(i["objects"]) for i in train)
    maximum_axis = max(max(o["mask"].shape) for i in train for o in i["objects"])
    for info in inspections["test"]:
        hollows = [o for o in info["objects"] if o["tile"] is not None]
        _require(hollows and all(o["tile"].tobytes() not in training_tiles for o in hollows), "test motif is not new")
        _require(len(info["objects"]) > maximum_count
                 or max(max(o["mask"].shape) for o in hollows) > maximum_axis, "missing count/scale shift")


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    _require(len(MASKS) == 178, "stencil space regression")
    shapes, stencils, silhouettes, palettes, layouts, count_vectors, axes = set(), set(), set(), set(), set(), set(), set()
    holes_seen, clipped, shared, open_bays = set(), 0, 0, 0
    regimes = set()
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        fingerprints.add(_fingerprint(puzzle))
        train_infos = [_analyze(p["input"]) for p in puzzle["train"]]
        regimes.add("count" if len(_analyze(puzzle["test"][0]["input"])["objects"]) > 4 else "scale")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list-input solve mismatch")
                shapes.add(pair["input"].shape)
                palettes.add((i["background"], i["foreground"]))
                layouts.add(_key(i["body"]))
                count_vectors.add((len(i["objects"]), sum(o["tile"] is not None for o in i["objects"])))
                clipped += bool(i["body"][0].any() or i["body"][-1].any() or i["body"][:, 0].any() or i["body"][:, -1].any())
                outlines = []
                for o in i["objects"]:
                    silhouettes.add(_key(o["mask"]))
                    axes.add(max(o["mask"].shape))
                    holes_seen.add(o["hole_count"])
                    open_bays += o["tile"] is None and _separated_runs(o["mask"])
                    if o["tile"] is not None:
                        stencils.add(o["tile"].tobytes())
                    footprint = np.zeros_like(i["body"])
                    footprint[o["rows"], o["cols"]] = True
                    outlines.append(_neighbors(footprint) & i["exterior"])
                shared += any(np.any(a & b) for k, a in enumerate(outlines) for b in outlines[k + 1:])
    _require(len(fingerprints) == 42 and len(layouts) >= 200, "layout diversity")
    _require(len(shapes) >= 40 and len(stencils) >= 70 and len(silhouettes) >= 150, "structural diversity")
    _require(len(palettes) >= 25 and len(count_vectors) >= 12 and len(axes) >= 12, "palette/count/scale diversity")
    _require(regimes == {"count", "scale"} and {0, 1, 2, 3, 4} <= holes_seen, "topology/shift diversity")
    _require(clipped >= 5 and shared >= 5 and open_bays >= 42, "boundary/bay diversity")
    # Every legal explicit count combination, using bounded small batches.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            validate_generated_puzzle(puzzle)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, nq), "count request ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Manual, source-independent geometry fixture: asymmetric L-shaped hole,
    # its open counterpart, a tiny solid, and an irregular multi-pocket motif.
    base = np.ones((9, 9), bool)
    l_tile = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], bool)
    base[2:5, 3:6] = ~l_tile
    opened = base.copy()
    opened[:3, 3] = False
    scene = np.zeros((13, 24), bool)
    scene[1:10, 1:10] = base
    scene[1:10, 12:21] = opened
    scene[12, 23] = True
    inp = np.where(scene, 1, 0).astype(np.int8)
    out = solve(inp)
    _require(np.all(out[3:6, 4:7][l_tile] == 6), "cavity fill regression")
    _require(out[9, 2] == (6 if l_tile[(8 + 1) % 3, (1 + 1) % 3] else 8), "remote phase regression")
    _require(out[9, 13] == 1 and out[3, 15] == 2, "open counterpart regression")
    _require(out[11, 22] == 2 and out[12, 23] == 1, "diagonal clipping regression")
    narrow_door = np.ones((9, 9), bool)
    narrow_door[2:7, 2:7] = False
    narrow_door[:3, 4] = False
    door_input = np.where(np.pad(narrow_door, 2), 1, 0).astype(np.int8)
    door_output = solve(door_input)
    _require(door_output[6, 6] == 0 and door_output[2, 6] == 2,
             "outline must not retroactively fill an open bay")
    four_pockets = np.ones((7, 11), bool)
    corner_tile = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], bool)
    four_pockets[2:5, 4:7] = ~corner_tile
    pocket_input = np.where(np.pad(four_pockets, 2), 7, 3).astype(np.int8)
    pocket_output = solve(pocket_input)
    _require(len(_regions(_holes(four_pockets))) == 4, "four-pocket topology regression")
    _require(pocket_output[2, 2:13].tolist() == [8] * 11
             and pocket_output[3, 2:13].tolist() == [8, 6, 6, 8, 6, 6, 8, 6, 6, 8, 6],
             "union stencil/empty middle row regression")
    _require(pocket_output[5, 7] == 8 and np.all(pocket_output[4:7, 6:9][corner_tile] == 6),
             "material bridges must not become cavity fill")
    translated = np.where(np.pad(four_pockets, ((3, 1), (1, 3))), 7, 3).astype(np.int8)
    _require(np.array_equal(solve(translated)[3:10, 1:12], pocket_output[2:9, 2:13]),
             "texture phase must move with its object")
    # Both omission-of-parent ablations must change returned training pixels
    # and must be rejected even with all color roles still present.
    for omission in ("A", "B"):
        puzzle = generate_puzzle(7, 2, 1)
        changed = False
        for pair in puzzle["train"]:
            info = _analyze(pair["input"])
            for o in info["objects"]:
                if omission == "A" and o["tile"] is None and _separated_runs(o["mask"]):
                    pair["output"][o["rows"], o["cols"]] = CARRIER
                    changed = True
                    break
                if omission == "B" and o["tile"] is not None:
                    rr, cc = o["rows"], o["cols"]
                    changed = bool(np.any(pair["output"][rr, cc] == ACCENT))
                    pair["output"][rr, cc] = CARRIER
                    break
            if changed:
                break
        _require(changed, "ablation had no effect")
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            pass
        else:
            raise AssertionError("parent-omission corruption accepted")
    return {"seeds": 42, "puzzles": len(fingerprints), "input_shapes": len(shapes),
            "stencils": len(stencils), "silhouettes": len(silhouettes),
            "input_palettes": len(palettes), "layouts": len(layouts),
            "count_class_combinations": len(count_vectors), "body_axis_sizes": sorted(axes),
            "cavity_counts": sorted(holes_seen), "clipped_scenes": clipped,
            "shared_outline_scenes": shared, "open_bay_solids": int(open_bays),
            "regimes": sorted(regimes), "independent_semantic_review": False}


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, topology and phase checks)")
