"""Champion blank windows, cavity decoration, and exterior halos.

No files, parent modules, global RNG, or training examples are consulted by solve.
All stochastic work is local to generate_puzzle. Experiments write no artifacts.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json

import numpy as np


HOLLOW, CAVITY, HALO, CORE = 8, 6, 2, 3
INPUT_COLORS = (0, 1, 4, 5, 7, 9)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _label(mask, neighbors=N4):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    regions = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        number = len(regions)
        labels[r, c] = number
        todo = deque([(r, c)])
        cells = []
        while todo:
            rr, cc = todo.popleft()
            cells.append((rr, cc))
            for dr, dc in neighbors:
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                        and labels[nr, nc] < 0):
                    labels[nr, nc] = number
                    todo.append((nr, nc))
        regions.append(cells)
    return labels, regions


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for dr, dc in N8:
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _rectangles(mask):
    """All blank rectangles nonextendable by a full row/column, inclusive ends.

    Integer bit rows keep the exhaustive O(H**2 W) search small on 30x30.
    Each contiguous common-column run is horizontally nonextendable already.
    """
    h, w = mask.shape
    rows = [sum(1 << c for c in range(w) if mask[r, c]) for r in range(h)]
    result = []
    for top in range(h):
        common = (1 << w) - 1
        for bottom in range(top, h):
            common &= rows[bottom]
            pending = common
            while pending:
                left = (pending & -pending).bit_length() - 1
                shifted = pending >> left
                length = (shifted ^ (shifted + 1)).bit_length() - 1
                span = ((1 << length) - 1) << left
                pending &= ~span
                if top and rows[top - 1] & span == span:
                    continue
                if bottom + 1 < h and rows[bottom + 1] & span == span:
                    continue
                result.append((top, left, bottom, left + length - 1))
    return result


def _area(rect):
    t, l, b, r = rect
    return (b - t + 1) * (r - l + 1)


def _square(rect):
    t, l, b, r = rect
    return b - t == r - l and b > t


def _analysis(grid):
    values, counts = np.unique(grid, return_counts=True)
    if (len(values) != 2 or int(counts.max()) * 2 <= grid.size
            or not set(map(int, values)) <= set(INPUT_COLORS)):
        raise ValueError("binary input needs a strict-majority background and reserved output colors")
    bg = int(values[np.argmax(counts)])
    fg = int(values[np.argmin(counts)])
    body, blank = grid == fg, grid == bg
    labels, objects = _label(body)
    blank_labels, regions = _label(blank)
    h, w = grid.shape
    boundary_ids = set(map(int, np.concatenate((blank_labels[0], blank_labels[-1],
                                              blank_labels[:, 0], blank_labels[:, -1])))) - {-1}
    exterior = np.isin(blank_labels, list(boundary_ids)) & blank
    cavities = blank & ~exterior
    hollow_ids, holes = set(), []
    for number, cells in enumerate(regions):
        if number in boundary_ids:
            continue
        owners = set()
        for rr, cc in cells:
            for dr, dc in N4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] >= 0:
                    owners.add(int(labels[nr, nc]))
        if len(owners) != 1:
            raise ValueError("nested or collectively owned cavity")
        owner = owners.pop()
        hollow_ids.add(owner)
        holes.append((owner, cells))
    hollow = np.isin(labels, list(hollow_ids)) & body
    halo = _dilate(body) & exterior
    rectangles = _rectangles(blank)
    biggest = max(map(_area, rectangles))
    maxima = [r for r in rectangles if _area(r) == biggest]
    champion = maxima[0] if len(maxima) == 1 and _square(maxima[0]) else None
    core = np.zeros_like(body)
    if champion is not None:
        t, l, b, r = champion
        core[t + 1:b, l + 1:r] = True
    return dict(background=bg, foreground=fg, body=body, labels=labels,
                objects=objects, holes=holes, hollow=hollow, cavities=cavities,
                exterior=exterior, halo=halo, rectangles=rectangles, maxima=maxima,
                champion=champion, core=core)


def _paint(grid, z, use_a=True, use_b=True):
    out = grid.copy()
    if use_b:
        out[z["hollow"]] = HOLLOW
        out[z["cavities"]] = CAVITY
    if use_a:
        out[z["core"]] = CORE
    if use_b:
        out[z["halo"]] = HALO
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive topology and the unique global champion solely from input pixels."""
    grid = _grid(input_grid)
    return _paint(grid, _analysis(grid))


def _no_diagonals(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _orient(rng, mask):
    mask = np.rot90(mask, int(rng.integers(4)))
    return np.fliplr(mask).copy() if rng.random() < .5 else mask.copy()


def _frame(rng, ch, cw, opened, irregular, multiple, fancy=True):
    walls = list(map(int, rng.integers(1, 3, size=4))) if fancy else [1] * 4
    if irregular:
        walls[int(rng.integers(4))] = 2
    top, bottom, left, right = walls
    h, w = ch + top + bottom, cw + left + right
    body = np.ones((h, w), dtype=bool)
    body[top:top + ch, left:left + cw] = False
    if irregular:
        # Partial side extensions preserve the complete original square window.
        eligible = [i for i, thickness in enumerate(walls) if thickness == 2]
        for side in rng.choice(eligible, size=int(rng.integers(1, len(eligible) + 1)), replace=False):
            if side < 2:
                start = left + int(rng.integers(1, cw - 1))
                length = int(rng.integers(1, cw - (start - left)))
                rr = top - 1 if side == 0 else top + ch
                body[rr, start:start + length] = False
            else:
                start = top + int(rng.integers(1, ch - 1))
                length = int(rng.integers(1, ch - (start - top)))
                cc = left - 1 if side == 2 else left + cw
                body[start:start + length, cc] = False
    if opened:
        door = left + int(rng.integers(1, cw - 1))
        body[:top, door] = False
    if fancy:
        # Outer corner cuts and broad partial tabs vary the body independently.
        if top == left == 2 and rng.random() < .5:
            body[0, 0] = False
        if bottom == right == 2 and rng.random() < .5:
            body[-1, -1] = False
        if rng.random() < .45:
            side = int(rng.integers(2))
            depth = int(rng.integers(1, 3))
            extended = np.pad(body, ((0, depth), (0, 0)))
            start = int(rng.integers(w - 1))
            stop = int(rng.integers(start + 2, w + 1))
            extended[h - 1:, start:stop] = True
            body = extended if side == 0 else np.rot90(extended, 2)
    if multiple:
        # A connected side lobe contains a separate 1x1 or 2x2 sealed pocket.
        mh = int(rng.integers(3, 5))
        mw = int(rng.integers(3, 5))
        h, w = body.shape
        rr = int(rng.integers(0, h - mh + 1))
        extended = np.pad(body, ((0, 0), (0, mw - 1)))
        extended[rr:rr + mh, w - 1:w + mw - 1] = True
        extended[rr + 1:rr + mh - 1, w:w + mw - 2] = False
        body = extended
    return _orient(rng, body)


def _pack_two(rng, masks, simple=False):
    # Tight but variable canvases let cavity and exterior windows compete in
    # the same global search. Assignment to left/right is randomized.
    a, b = masks if rng.random() < .5 else masks[::-1]
    margin = 1 if simple else int(rng.integers(0, 2))
    gap = 1 if simple else int(rng.integers(1, 4))
    h = max(a.shape[0], b.shape[0]) + 2 * margin
    w = a.shape[1] + b.shape[1] + gap + 2 * margin
    if max(h, w) > 30:
        return None
    canvas = np.zeros((h, w), bool)
    ar = margin if simple else int(rng.integers(0, h - a.shape[0] + 1))
    br = margin if simple else int(rng.integers(0, h - b.shape[0] + 1))
    canvas[ar:ar + a.shape[0], margin:margin + a.shape[1]] = a
    bc = margin + a.shape[1] + gap
    canvas[br:br + b.shape[0], bc:bc + b.shape[1]] = b
    return _orient(rng, canvas)


def _matches(z, mode, side):
    if mode == "closed":
        return (z["champion"] is not None and _area(z["champion"]) == side * side
                and z["core"].any() and np.all(z["cavities"][z["core"]]))
    if mode == "open":
        return (z["champion"] is not None and _area(z["champion"]) == side * side
                and z["core"].any() and np.all(z["exterior"][z["core"]]))
    if mode == "nonsquare":
        return len(z["maxima"]) == 1 and not _square(z["maxima"][0])
    return len(z["maxima"]) == 2 and all(_square(r) for r in z["maxima"])


def _add_small_objects(rng, body, bg, fg, count):
    # Optional bodies use only exterior cells, never another object's cavity.
    for _ in range(count):
        z = _analysis(np.where(body, fg, bg).astype(np.int8))
        for _ in range(24):
            style = int(rng.integers(4))
            if style == 0:
                small = np.ones((1, int(rng.integers(1, 5))), bool)
            elif style == 1:
                small = np.array([[1, 0], [1, 1]], bool)
            elif style == 2:
                small = np.ones((int(rng.integers(1, 4)), int(rng.integers(1, 4))), bool)
            else:
                small = np.ones((3, 3), bool)
                small[1, 1] = False
            small = _orient(rng, small)
            h, w = small.shape
            if h > body.shape[0] or w > body.shape[1]:
                continue
            r = int(rng.integers(body.shape[0] - h + 1))
            c = int(rng.integers(body.shape[1] - w + 1))
            free = z["exterior"] & ~_dilate(body)
            # Requiring the whole box exterior excludes nested tiny rings.
            if not free[r:r + h, c:c + w].all():
                continue
            candidate = body.copy()
            candidate[r:r + h, c:c + w] |= small
            if int(candidate.sum()) * 2 >= candidate.size or not _no_diagonals(candidate):
                continue
            try:
                _analysis(np.where(candidate, fg, bg).astype(np.int8))
            except ValueError:
                continue
            body = candidate
            break
    return body


def _make_pair(rng, mode, bg, fg, irregular, multiple, extra, target_side=None):
    # Retry geometry in memory. Final simple attempts retain every teaching
    # feature (including irregular/multiple cavities) rather than dropping it.
    for attempt in range(160):
        fancy = attempt < 100
        side = int(rng.integers(5, 9)) if target_side is None else target_side
        if attempt >= 140 and target_side is not None and target_side >= 8:
            side = 7
        other = side if mode == "tie" else side - 1
        main_closed = mode != "open"
        main_dims = (side, side + 1) if mode == "nonsquare" else (side, side)
        first = _frame(rng, *main_dims, not main_closed,
                       irregular and main_closed, multiple and main_closed, fancy)
        second = _frame(rng, other, other, main_closed,
                        irregular and not main_closed, multiple and not main_closed, fancy)
        body = _pack_two(rng, [first, second], simple=not fancy)
        if (body is None or not _no_diagonals(body)
                or int(body.sum()) * 2 >= body.size):
            continue
        inp = np.where(body, fg, bg).astype(np.int8)
        try:
            z = _analysis(inp)
        except ValueError:
            continue
        if not _matches(z, mode, side):
            continue
        # Every scene contrasts an original open bay with an enclosed cavity.
        if not (z["hollow"].any() and (z["body"] & ~z["hollow"]).any()):
            continue
        if extra:
            body = _add_small_objects(rng, body.copy(), bg, fg, extra)
            inp = np.where(body, fg, bg).astype(np.int8)
            z = _analysis(inp)
        if not _matches(z, mode, side) or not _no_diagonals(body):
            continue
        if sum(_square(r) for r in z["rectangles"]) < (1 if mode == "nonsquare" else 2):
            continue
        return {"input": inp, "output": _paint(inp, z)}
    raise RuntimeError("bounded scene construction exhausted")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    bg, fg = map(int, rng.choice(INPUT_COLORS, size=2, replace=False))
    irregular = bool(rng.random() < .65)
    multiple = bool(rng.random() < .35)
    train_modes = ["closed", "open"]
    if nt >= 3:
        train_modes.append("nonsquare")
    if nt >= 4:
        train_modes.append("tie")
    available = train_modes.copy()
    train_modes += [str(rng.choice(available)) for _ in range(nt - len(train_modes))]
    # The first two contrasting topology examples remain readable anchors.
    # Tests recombine familiar modes on the larger side band with more small
    # objects, rather than introducing an unshown tie or non-square rejection.
    puzzle, seen = {"train": [], "test": []}, set()
    for split, modes in (("train", train_modes),
                         ("test", [str(rng.choice(available)) for _ in range(nq)])):
        for i, mode in enumerate(modes):
            if split == "train" and i == 0:
                side = int(rng.integers(5, 7))
            elif split == "train" and i == 1:
                side = 7
            else:
                side = int(rng.integers(5, 8)) if split == "train" else int(rng.integers(7, 10))
            extra = int(rng.integers(0, 2)) if split == "train" else int(rng.integers(1, 4))
            for _ in range(20):
                pair = _make_pair(rng, mode, bg, fg, irregular, multiple, extra, side)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded input uniqueness attempts exhausted")
    return puzzle


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _features(grid, z=None):
    z = _analysis(grid) if z is None else z
    holes_per_object = [sum(owner == i for owner, cells in z["holes"])
                        for i in range(len(z["objects"]))]
    irregular = 0
    for owner, cells in z["holes"]:
        rr, cc = np.array(cells).T
        irregular += len(cells) != (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
    if z["champion"] is not None:
        mode = "closed" if np.any(z["core"] & z["cavities"]) else "open"
    else:
        mode = "nonsquare" if len(z["maxima"]) == 1 else "tie"
    return dict(mode=mode, objects=len(z["objects"]), holes=len(z["holes"]),
                hollow_objects=sum(n > 0 for n in holes_per_object),
                multi=any(n > 1 for n in holes_per_object), irregular=int(irregular),
                core_cells=int(z["core"].sum()), halo_cells=int(z["halo"].sum()),
                max_area=_area(z["maxima"][0]), maxima=len(z["maxima"]),
                local_squares=sum(_square(r) for r in z["rectangles"]),
                clipped=bool(z["body"][0].any() or z["body"][-1].any()
                             or z["body"][:, 0].any() or z["body"][:, -1].any()))


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, palettes, train_modes, test_modes = set(), set(), set(), set()
    train_features, test_features = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(inp.shape == out.shape, "dimensions must be preserved")
            key = (inp.shape, inp.tobytes())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            z = _analysis(inp)
            palettes.add((z["background"], z["foreground"]))
            _check(_no_diagonals(z["body"]), "ambiguous diagonal contact")
            _check(len(_label(z["body"], N8)[1]) == len(z["objects"]), "diagonal object contact")
            # Four/eight-connected blank partitions agree, including open bays.
            labels8, regions8 = _label(~z["body"], N8)
            _check(len(regions8) == len(_label(~z["body"])[1]), "diagonal blank leak")
            _check(z["hollow"].any() and (z["body"] & ~z["hollow"]).any(), "topology contrast absent")
            _check(z["halo"].any(), "halo absent")
            _check(not np.any(z["core"] & (z["body"] | z["halo"])), "strict-interior geometry")
            _check(np.all(out[z["body"] & ~z["hollow"]] == z["foreground"]), "solid body changed")
            _check(np.all(out[z["hollow"]] == HOLLOW), "incomplete whole hollow-body recoloring")
            _check(np.all(out[z["cavities"] & ~z["core"]] == CAVITY), "cavity collar/fill")
            _check(np.all(out[z["core"]] == CORE), "champion strict interior")
            _check(np.all(out[z["halo"]] == HALO), "exterior eight-neighbor halo")
            untouched = z["exterior"] & ~z["core"] & ~z["halo"]
            _check(np.all(out[untouched] == z["background"]), "distant exterior changed")
            _check(np.array_equal(out, solve(inp)), "solve mismatch")
            f = _features(inp, z)
            _check(f["local_squares"] >= (2 if f["mode"] != "nonsquare" else 1),
                   "global competition not visible")
            (train_modes if split == "train" else test_modes).add(f["mode"])
            (train_features if split == "train" else test_features).append(f)
            if z["core"].any():
                _check(not np.array_equal(out, _paint(inp, z, use_a=False)), "A is inert")
            _check(not np.array_equal(out, _paint(inp, z, use_b=False)), "B is inert")
    _check(len(palettes) == 1, "input color roles vary within puzzle")
    _check({"closed", "open"} <= train_modes, "missing open/closed winner demonstrations")
    _check(test_modes <= train_modes, "unshown global-selection mode in test")
    for feature in ("irregular", "multi"):
        _check(not any(f[feature] for f in test_features) or any(f[feature] for f in train_features),
               "test cavity extension not demonstrated: " + feature)


def _fingerprint(puzzle):
    return hashlib.sha256(b"".join(str(p["input"].shape).encode() + p["input"].tobytes()
                                  + p["output"].tobytes() for s in ("train", "test")
                                  for p in puzzle[s])).hexdigest()


def _regressions():
    # Literal fixtures specify expected pixels rather than merely round-trip
    # labels. A one-cell door changes topology but leaves the champion square.
    body = np.ones((7, 7), bool)
    body[1:6, 1:6] = False
    closed = np.where(np.pad(body, 1), 1, 0).astype(np.int8)
    opened = closed.copy()
    opened[1, 4] = 0
    a, b = solve(closed), solve(opened)
    _check(a[4, 4] == b[4, 4] == CORE, "same blank champion across a door toggle")
    _check(a[2, 4] == CAVITY and b[2, 4] == HALO, "topological collar interaction")
    _check(a[1, 3] == HOLLOW and b[1, 3] == 1, "body topology classification")
    _check(a[0, 0] == HALO, "diagonal halo")
    _check(np.count_nonzero(a == CORE) == 9, "strict interior, not all 25 square cells")
    # A remote rival changes the first chamber's core without changing its
    # topology: unique square -> equal squares -> larger non-square.
    tied = np.zeros((9, 17), np.int8)
    tied[1:8, 1:8] = body
    tied[1:8, 9:16] = body
    unique = tied.copy()
    unique[2:7, 14] = 1
    _check(np.count_nonzero(solve(unique) == CORE) == 9, "unique global winner")
    _check(not np.any(solve(tied) == CORE), "remote equal-area suppression")
    nonsquare = np.zeros((9, 18), np.int8)
    nonsquare[1:8, 1:8] = body
    other = np.ones((7, 8), bool)
    other[1:6, 1:7] = False
    nonsquare[1:8, 9:17] = other
    _check(not np.any(solve(nonsquare) == CORE), "rank non-squares before filtering")
    # A 2x2 champion is valid but has an empty strict interior.
    tiny_blank = np.zeros((4, 4), bool)
    tiny_blank[1:3, 1:3] = True
    _check(_rectangles(tiny_blank) == [(1, 1, 2, 2)], "2x2 square eligibility")
    tiny_core = np.zeros_like(tiny_blank)
    tiny_core[2:2, 2:2] = True
    _check(not tiny_core.any(), "2x2 empty strict interior")
    # Cross-check the rectangle primitive against full rectangle enumeration.
    rng = np.random.default_rng(3207)
    for _ in range(24):
        mask = rng.random((5, 6)) < .6
        expected = []
        for t in range(5):
            for l in range(6):
                for b in range(t, 5):
                    for r in range(l, 6):
                        if not mask[t:b + 1, l:r + 1].all():
                            continue
                        extend = ((t > 0 and mask[t - 1, l:r + 1].all())
                                  or (b < 4 and mask[b + 1, l:r + 1].all())
                                  or (l > 0 and mask[t:b + 1, l - 1].all())
                                  or (r < 5 and mask[t:b + 1, r + 1].all()))
                        if not extend:
                            expected.append((t, l, b, r))
        _check(set(expected) == set(_rectangles(mask)), "rectangle enumeration regression")


def self_test():
    """Bounded 42-seed sweep; retains statistics, not all puzzle payloads."""
    _regressions()
    dimensions, structures, palettes, fingerprints = set(), set(), set(), set()
    modes, object_counts, champion_sides = set(), set(), set()
    totals = dict(pairs=0, irregular=0, multi=0, clipped=0, closed=0, open=0, nonsquare=0, tie=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = _fingerprint(puzzle)
        _check(fingerprint == _fingerprint(generate_puzzle(seed)), "seed nondeterminism")
        fingerprints.add(fingerprint)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                z = _analysis(g)
                f = _features(g, z)
                dimensions.add(g.shape)
                structures.add((g.shape, (g == z["foreground"]).tobytes()))
                palettes.add((z["background"], z["foreground"]))
                object_counts.add(f["objects"])
                modes.add(f["mode"])
                if z["champion"] is not None:
                    champion_sides.add(z["champion"][2] - z["champion"][0] + 1)
                totals["pairs"] += 1
                totals[f["mode"]] += 1
                for key in ("irregular", "multi", "clipped"):
                    totals[key] += int(bool(f[key]))
    _check(len(fingerprints) == 42 and len(structures) >= 180, "structural repetition")
    _check(len(dimensions) >= 35 and len(palettes) >= 15, "canvas/palette diversity")
    _check(len(champion_sides) >= 4 and len(object_counts) >= 3, "size/count diversity")
    _check(modes == {"closed", "open", "nonsquare", "tie"}, "selection diversity")
    _check(totals["irregular"] >= 20 and totals["multi"] >= 10 and totals["clipped"] >= 20,
           "topology/body diversity")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(319, nt, nq)
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject a corrupted zone, even when all colors remain represented.
    bad = generate_puzzle(17)
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), grid_shapes=len(dimensions),
                normalized_structures=len(structures), input_palettes=len(palettes),
                object_counts=sorted(object_counts), champion_sides=sorted(champion_sides), **totals)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
