"""Perforated stamps: enclosed negative space prescribes positive replicas.

Only Python and NumPy are used.  The solver reads one input and has no access
to a seed, training outputs, sampler decisions, files, or mutable puzzle state.
"""
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(body, neighbors=N4):
    """Background unreachable from outside an isolated stamp."""
    a = np.pad(np.asarray(body, dtype=bool), 1)
    visited = np.zeros_like(a)
    visited[0, 0] = True
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            y, x = r + dr, c + dc
            if (0 <= y < a.shape[0] and 0 <= x < a.shape[1]
                    and not a[y, x] and not visited[y, x]):
                visited[y, x] = True
                stack.append((y, x))
    return (~a & ~visited)[1:-1, 1:-1]


def _crop(mask):
    y, x = np.nonzero(mask)
    _require(len(y) > 0, "cannot crop an empty mask")
    return mask[y.min():y.max()+1, x.min():x.max()+1].copy()


def _clear_topology(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _stamp_ok(body):
    if not body.any() or len(_components(body)) != 1 or not _clear_topology(body):
        return False
    holes = _holes(body)
    if int(holes.sum()) < 3 or not np.array_equal(holes, _holes(body, N8)):
        return False
    if len(_components(holes)) != len(_components(holes, N8)):
        return False
    h, w = _crop(holes).shape
    return (body.shape[0]+1)*h-1 <= 28 and (body.shape[1]+1)*w-1 <= 28


def _registrations(stencil, positions, pitch, shape=None, stamp_shape=None):
    """Every translation containing ALL clues, not a maximum-overlap heuristic.

    Without shape, this checks registration on the infinite implicit lattice.
    Thus an accepted fragment does not rely on blank canvas margins as clues.
    """
    active = [tuple(map(int, p)) for p in np.argwhere(stencil)]
    sy, sx = pitch
    p0 = positions[0]
    origins = set()
    for y, x in active:
        origin = p0[0]-sy*y, p0[1]-sx*x
        complete = {(origin[0]+sy*r, origin[1]+sx*c) for r, c in active}
        if not set(positions) <= complete:
            continue
        if shape is not None:
            bh, bw = stamp_shape
            if not all(1 <= r and 1 <= c and r+bh < shape[0] and c+bw < shape[1]
                       for r, c in complete):
                continue
        origins.add(origin)
    return sorted(origins)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid ARC color")
    return a


def _read_input(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "uniform background frame required")
    colors = set(map(int, np.unique(a)))
    _require(len(colors) == 2, "input must be binary")
    fg = (colors - {bg}).pop()
    regions = _components(a == fg)
    _require(len(regions) >= 2, "at least two supplied stamps required")
    template, positions = None, []
    for cells in regions:
        rr, cc = np.array(cells).T
        y, x = int(rr.min()), int(cc.min())
        local = np.zeros((int(rr.max())-y+1, int(cc.max())-x+1), bool)
        local[rr-y, cc-x] = True
        if template is None:
            template = local
        _require(np.array_equal(local, template), "all supplied stamps must be identical")
        positions.append((y, x))
    _require(_stamp_ok(template), "invalid stamp topology or footprint")
    bh, bw = template.shape
    # Requiring blank disjoint bounding boxes makes whole-object copying exact,
    # even for an irregular silhouette with an exterior bay.
    for y, x in positions:
        _require(np.array_equal(a[y:y+bh, x:x+bw] == fg, template), "interlocking stamps")
    holes = _holes(template)
    stencil = _crop(holes)
    pitch = bh+1, bw+1
    origins = _registrations(stencil, positions, pitch)
    _require(len(origins) == 1, "clues must have one boundary-independent registration")
    origin = origins[0]
    _require(_registrations(stencil, positions, pitch, a.shape, template.shape) == [origin],
             "completed pattern must fit inside the frame")
    active = {(origin[0]+int(r)*pitch[0], origin[1]+int(c)*pitch[1])
              for r, c in np.argwhere(stencil)}
    _require(set(positions) < active, "a strict partial stencil is required")
    return {"background": bg, "foreground": fg, "body": template, "holes": holes,
            "stencil": stencil, "pitch": pitch, "origin": origin,
            "positions": positions, "complete": active, "missing": active-set(positions)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = np.asarray(input_grid)
    f = _read_input(a)
    out = a.copy()
    h, w = f["body"].shape
    stamp = np.where(f["body"], f["foreground"], f["background"])
    for y, x in sorted(f["missing"]):
        out[y:y+h, x:x+w] = stamp
    return out


def _fragment(rng, stencil):
    """Random strict fragments; bounded fallback searches all leave-one-outs."""
    points = [tuple(map(int, p)) for p in np.argwhere(stencil)]
    n = len(points)
    # Registration is in coarse units here; this is only a sampler helper.
    for _ in range(96):
        count = int(rng.integers(2, n))
        chosen = sorted(map(int, rng.choice(n, count, replace=False)))
        fragment = [points[i] for i in chosen]
        if _registrations(stencil, fragment, (1, 1)) == [(0, 0)]:
            return fragment
    for missing in rng.permutation(n):
        fragment = [p for i, p in enumerate(points) if i != int(missing)]
        if _registrations(stencil, fragment, (1, 1)) == [(0, 0)]:
            return fragment
    return None


def _orient(mask, rotation, flip):
    a = np.rot90(mask, rotation)
    return (np.fliplr(a) if flip else a).copy()


def _teaching_stamps(rng):
    """One plugged doorway changes a bay into a second enclosed region."""
    h, w = int(rng.integers(5, 7)), 6
    opened = np.ones((h, w), bool)
    opened[1:h-1, 1] = False
    foot = h-2 if rng.integers(2) else 1
    opened[foot, 2] = False       # An irregular L cavity.
    opened[:h-1, 4] = False      # A deep exterior bay.
    closed = opened.copy()
    closed[1, 4] = True          # Plug one cell; top cell remains an open notch.
    # The same optional corner cuts affect both silhouettes.
    for y, x in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)):
        if rng.random() < .35:
            p, q = opened.copy(), closed.copy()
            p[y, x] = q[y, x] = False
            if _stamp_ok(p) and _stamp_ok(q):
                opened, closed = p, q
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    return _orient(opened, k, flip), _orient(closed, k, flip)


def _random_stamp(rng):
    h, w = map(int, rng.integers(4, 8, size=2))
    max_h = min(h-2, 29//(h+1))
    max_w = min(w-2, 29//(w+1))
    kind = int(rng.integers(6))
    ch, cw = int(rng.integers(1, max_h+1)), int(rng.integers(1, max_w+1))
    if kind == 4 and max_w >= 3:
        cw = int(rng.integers(3, max_w+1))
        ch = int(rng.integers(2, max_h+1))
    if kind == 5 and max_h >= 3 and max_w >= 3:
        ch, cw = int(rng.integers(3, max_h+1)), int(rng.integers(3, max_w+1))
    if ch*cw < 3:
        return None
    r = int(rng.integers(1, h-ch))
    c = int(rng.integers(1, w-cw))
    cavity = np.zeros((ch, cw), bool)
    if kind == 0:
        cavity[:] = True
    elif kind == 1:
        # Connected bent pockets; transpose/reflection happen below.
        cavity[:, 0] = cavity[-1, :] = True
    elif kind == 4 and cw >= 3:
        # Independently sized chambers separated by a genuine wall. With four
        # columns, one chamber can be bent while the other remains a narrow
        # pocket. These combine multiple cavities and irregular geometry.
        left = cw-2
        cavity[:, 0] = True
        cavity[-1, :left] = True
        start = int(rng.integers(ch))
        cavity[start:, -1] = True
    elif kind == 5 and ch >= 3 and cw >= 3:
        # Three or four pockets, optionally elongated independently. They are
        # separated by body cells, not diagonal contacts or a shared cavity.
        corners = ((0, 0), (0, cw-1), (ch-1, 0), (ch-1, cw-1))
        n = int(rng.integers(3, 5))
        for i in rng.choice(4, n, replace=False):
            y, x = corners[int(i)]
            cavity[y, x] = True
            if ch == 4 and rng.random() < .5:
                cavity[1 if y == 0 else ch-2, x] = True
    else:
        n = int(rng.integers(3, min(ch*cw, 12)+1))
        cavity.flat[rng.choice(ch*cw, n, replace=False)] = True
    body = np.ones((h, w), bool)
    body[r:r+ch, c:c+cw] &= ~cavity
    if not _stamp_ok(body):
        return None
    original_holes = _holes(body)
    # Independent sculpting creates open bays, steps, variable walls and lobes.
    for _ in range(int(rng.integers(3, 16))):
        trial = body.copy()
        side = int(rng.integers(4))
        if side < 2:
            x = int(rng.integers(w))
            depth = int(rng.integers(1, h))
            if side == 0:
                trial[:depth, x] = False
            else:
                trial[h-depth:, x] = False
        else:
            y = int(rng.integers(h))
            depth = int(rng.integers(1, w))
            if side == 2:
                trial[y, :depth] = False
            else:
                trial[y, w-depth:] = False
        if _stamp_ok(trial) and np.array_equal(_holes(trial), original_holes):
            body = trial
    body = _crop(body)
    return _orient(body, int(rng.integers(4)), bool(rng.integers(2)))


def _assemble(rng, body, palette, fragment=None):
    stencil = _crop(_holes(body))
    fragment = _fragment(rng, stencil) if fragment is None else fragment
    if fragment is None:
        return None
    bh, bw = body.shape
    sh, sw = stencil.shape
    extent = (bh+1)*sh-1, (bw+1)*sw-1
    margins = []
    for size in extent:
        total = int(rng.integers(2, min(30-size, 10)+1))
        before = int(rng.integers(1, total))
        margins.append((before, total-before))
    top, bottom = margins[0]
    left, right = margins[1]
    shape = extent[0]+top+bottom, extent[1]+left+right
    bg, fg = palette
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    stamp = np.where(body, fg, bg).astype(np.int8)
    for y, x in np.argwhere(stencil):
        r, c = top+int(y)*(bh+1), left+int(x)*(bw+1)
        out[r:r+bh, c:c+bw] = stamp
    for y, x in fragment:
        r, c = top+y*(bh+1), left+x*(bw+1)
        inp[r:r+bh, c:c+bw] = stamp
    return {"input": inp, "output": out}


def _check_count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _check_count(num_train, "num_train", 2, 10)
    nq = _check_count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    anchors = _teaching_stamps(rng)
    base_palette = tuple(map(int, rng.choice(10, 2, replace=False)))
    stable_palette = bool(rng.random() < .35)
    puzzle, seen, trained_masks = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            # The near-matched teaching contrast retains colors in its two pairs.
            palette = (base_palette if stable_palette or (split == "train" and index < 2)
                       else tuple(map(int, rng.choice(10, 2, replace=False))))
            pair = None
            for _ in range(384):
                body = anchors[index] if split == "train" and index < 2 else _random_stamp(rng)
                if body is None:
                    continue
                stencil_key = _design_key(_crop(_holes(body)))
                if split == "test" and stencil_key in trained_masks:
                    continue
                candidate = _assemble(rng, body, palette)
                if candidate is not None and _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            if pair is None:
                # Finite fallback: scan the 3x3 cavity vocabulary inside a 5x5
                # body. This offers many more stencils than ten demonstrations.
                for code in rng.permutation(np.arange(1, 512)):
                    body = np.ones((5, 5), bool)
                    cavity = np.array([(int(code) >> i) & 1 for i in range(9)], bool).reshape(3, 3)
                    body[1:4, 1:4] &= ~cavity
                    if not _stamp_ok(body):
                        continue
                    stencil_key = _design_key(_crop(_holes(body)))
                    if split == "test" and stencil_key in trained_masks:
                        continue
                    candidate = _assemble(rng, body, palette)
                    if candidate is not None and _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            _require(pair is not None, "finite stamp construction exhausted")
            seen.add(_key(pair["input"]))
            if split == "train":
                trained_masks.add(stencil_key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _symmetries(mask):
    transforms = [np.rot90(mask, k) for k in range(4)]
    transforms += [np.fliplr(a) for a in transforms]
    return {i for i, a in enumerate(transforms) if _key(_crop(a)) == _key(mask)}


def _design_key(mask):
    """Identify a design up to translation, rotation and reflection."""
    transforms = [np.rot90(mask, k) for k in range(4)]
    transforms += [np.fliplr(a) for a in transforms]
    return min(_key(_crop(a)) for a in transforms)


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "output dimensions changed")
    f = _read_input(inp)
    bg, fg, body = f["background"], f["foreground"], f["body"]
    bh, bw = body.shape
    # Validate by reconstructing the complete component arrangement, rather than
    # calling solve or trusting an output produced by generation.
    expected = np.full(inp.shape, bg, dtype=np.int8)
    for y, x in f["complete"]:
        expected[y:y+bh, x:x+bw][body] = fg
    _require(np.array_equal(expected, out), "output is not the exact hole-stencil arrangement")
    _require(np.all(out[inp == fg] == fg), "supplied foreground changed")
    _require(len(_components(out == fg)) == int(f["stencil"].sum()), "wrong replica count")
    for y, x in f["positions"]:
        _require(np.array_equal(inp[y:y+bh, x:x+bw], out[y:y+bh, x:x+bw]),
                 "existing stamp or its cavity was changed")
    for y, x in f["missing"]:
        _require(np.all(inp[y:y+bh, x:x+bw] == bg), "target block is not blank")
        _require(np.all(out[y:y+bh, x:x+bw][f["holes"]] == bg), "replicas must retain their holes")
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            f = _inspect_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            features[split].append(f)
    train = features["train"]
    _require(any(np.any(~f["body"] & ~f["holes"]) for f in train), "no open-background contrast")
    _require(any(len(_components(f["holes"])) >= 2 for f in train), "no multiple-cavity demonstration")
    _require(any(any(len(c) < _crop_region_area(c) for c in _components(f["holes"])) for f in train),
             "no irregular-cavity demonstration")
    symmetries = set(range(8))
    for f in train:
        symmetries &= _symmetries(f["stencil"])
    _require(symmetries == {0}, "training does not establish unchanged orientation")
    train_keys = {_design_key(f["stencil"]) for f in train}
    _require(len(train_keys) >= 2, "training stencil must vary")
    _require(all(_design_key(f["stencil"]) not in train_keys for f in features["test"]),
             "test should transfer to unseen negative-space stencils")


def _crop_region_area(cells):
    rows, cols = zip(*cells)
    return (max(rows)-min(rows)+1)*(max(cols)-min(cols)+1)


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def variation_report(puzzles):
    records = [_inspect_pair(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    return {
        "puzzles": len(puzzles), "pairs": len(records),
        "canvas_shapes": len({np.asarray(pair["input"]).shape for p in puzzles for s in p for pair in p[s]}),
        "stamp_shapes": len({f["body"].shape for f in records}),
        "stamp_silhouettes": len({_key(f["body"]) for f in records}),
        "cavity_stencils": len({_key(f["stencil"]) for f in records}),
        "cavity_design_classes": len({_design_key(f["stencil"]) for f in records}),
        "palettes": len({(f["background"], f["foreground"]) for f in records}),
        "cavity_counts": sorted({len(_components(f["holes"])) for f in records}),
        "replica_counts": sorted({len(f["complete"]) for f in records}),
        "supplied_counts": sorted({len(f["positions"]) for f in records}),
        "missing_counts": sorted({len(f["missing"]) for f in records}),
        "open_bay_pairs": sum(bool(np.any(~f["body"] & ~f["holes"])) for f in records),
        "asymmetric_stencils": sum(_symmetries(f["stencil"]) == {0} for f in records),
        "one_missing_pairs": sum(len(f["missing"]) == 1 for f in records),
        "less_than_half_supplied": sum(2*len(f["positions"]) < len(f["complete"]) for f in records),
    }


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    report = variation_report(puzzles)
    _require(report["canvas_shapes"] >= 40, "insufficient canvas variation")
    _require(report["stamp_silhouettes"] >= 80, "insufficient source shape variation")
    _require(report["cavity_stencils"] >= 35, "insufficient negative-space variation")
    _require(report["palettes"] >= 25, "insufficient palette variation")
    _require(len(report["replica_counts"]) >= 5 and len(report["missing_counts"]) >= 4,
             "insufficient replica/omission variation")
    _require(report["one_missing_pairs"] >= 10 and report["less_than_half_supplied"] >= 5,
             "insufficient fragment density variation")
    test_features = [_read_input(pair["input"]) for p in puzzles for pair in p["test"]]
    _require(any(len(_components(f["holes"])) >= 3 for f in test_features),
             "multiple-cavity counts do not generalize")
    _require(any(len(_components(f["holes"])) >= 2
                 and any(len(c) < _crop_region_area(c) for c in _components(f["holes"]))
                 for f in test_features), "irregular and multiple cavities do not combine")
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                copied = pair["input"].copy()
                predicted = solve(copied)
                _require(np.array_equal(predicted, pair["output"]), "input-only solve mismatch")
                _require(np.array_equal(copied, pair["input"]), "solve mutated input")
        # Both inherited mechanisms have observable counterfactual witnesses.
        f = _read_input(puzzle["train"][0]["input"])
        closed = _read_input(puzzle["train"][1]["input"])
        _require(f["body"].shape == closed["body"].shape
                 and np.count_nonzero(f["body"] != closed["body"]) == 1,
                 "the doorway contrast must differ in exactly one body cell")
        _require(np.all(~f["holes"] | closed["holes"])
                 and int(closed["holes"].sum()) > int(f["holes"].sum()),
                 "plugging the doorway must create enclosed space")
        _require(_key(_crop(~f["body"])) != _key(f["stencil"]), "A ablation has no witness")
        _require(_key(f["body"]) != _key(f["stencil"]), "ink and cavity stencils coincide")
        for pair in puzzle["train"]:
            f = _read_input(pair["input"])
            only_fill = pair["input"].copy()
            for y, x in f["positions"]:
                view = only_fill[y:y+f["body"].shape[0], x:x+f["body"].shape[1]]
                view[f["holes"]] = f["foreground"]
            _require(not np.array_equal(only_fill, pair["output"]), "B ablation has no witness")
    _require(_same(puzzles[0], generate_puzzle(0)), "seed is not deterministic")
    _require(not _same(puzzles[0], puzzles[1]), "cross-seed repetition")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(73, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts not honored")
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(7, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # Registration must reject equal-overlap ambiguity, even if the canvas is
    # large enough to contain either answer.
    rect = np.ones((3, 3), bool)
    _require(len(_registrations(rect, [(0, 0), (1, 0)], (1, 1))) > 1,
             "ambiguous fragment was accepted")
    # At a fixed mask, fragment diversity is genuine, not only palette changes.
    mask = np.array([[1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1]], bool)
    fragments = {tuple(_fragment(np.random.default_rng(s), mask)) for s in range(42)}
    _require(len(fragments) >= 12, "fixed-stencil clues do not vary")
    # Reject missing replicas, filling an original cavity, altering a bay,
    # changing a gutter, and damage to an existing body.
    original = puzzles[0]
    f = _read_input(original["train"][0]["input"])
    bh, bw = f["body"].shape
    y, x = f["positions"][0]
    hole = np.argwhere(f["holes"])[0]
    bay = np.argwhere(~f["body"] & ~f["holes"])[0]
    body_cell = np.argwhere(f["body"])[0]
    my, mx = min(f["missing"])
    corruptions = [(y+int(hole[0]), x+int(hole[1])),
                   (y+int(bay[0]), x+int(bay[1])),
                   (y+int(body_cell[0]), x+int(body_cell[1])),
                   (my+int(body_cell[0]), mx+int(body_cell[1])), (0, 0)]
    for r, c in corruptions:
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
        grid = bad["train"][0]["output"]
        grid[r, c] = (int(grid[r, c])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted semantic corruption")
