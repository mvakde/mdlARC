"""Compass-oriented keyed masks, combined by odd/even coverage.

Only standard Python and NumPy are used. The parser and solver reconstruct all
roles, reference offsets and transformations from the supplied input.
"""
from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("invalid ARC grid")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _transform(cells, t):
    result = {}
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result[int(r), int(c)] = int(v)
    return result


def _signature(cells):
    return tuple(sorted((r, c, v) for (r, c), v in cells.items()))


def _bounds(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _direction(p):
    r, c = p
    if r == 0 and c:
        return 1 if c > 0 else 3
    if c == 0 and r:
        return 2 if r > 0 else 0
    raise ValueError("compass terminal is not on an anchor axis")


def _board_candidates(g):
    """Straight colored sides, blank corners, and a uniform inner rim."""
    bg = int(g[0, 0])
    h, w = g.shape
    result = []
    for top in range(h - 8):
        c = 0
        while c < w:
            start, color = c, int(g[top, c])
            while c < w and g[top, c] == color:
                c += 1
            if color == bg or c - start < 7 or start == 0 or c == w:
                continue
            left, right = start - 1, c
            if g[top, left] != bg or g[top, right] != bg:
                continue
            for bottom in range(top + 8, h):
                south = int(g[bottom, start])
                if south in (bg, color) or not np.all(g[bottom, start:right] == south):
                    continue
                if g[bottom, left] != bg or g[bottom, right] != bg:
                    continue
                west, east = int(g[top + 1, left]), int(g[top + 1, right])
                bands = (color, east, south, west)
                if len(set(bands)) != 4 or bg in bands:
                    continue
                if (not np.all(g[top + 1:bottom, left] == west)
                        or not np.all(g[top + 1:bottom, right] == east)):
                    continue
                inner = g[top + 1:bottom, left + 1:right]
                fill = int(inner[0, 0])
                if fill in bands:
                    continue
                if not (np.all(inner[0] == fill) and np.all(inner[-1] == fill)
                        and np.all(inner[:, 0] == fill) and np.all(inner[:, -1] == fill)):
                    continue
                anchors = [(int(r) + 1, int(c1) + 1, int(inner[r, c1]))
                           for r, c1 in zip(*np.nonzero(inner != fill))]
                keys = {k for _, _, k in anchors}
                if not (1 <= len(keys) <= 3) or keys & (set(bands) | {bg}):
                    continue
                points = {(r, c1) for r, c1, _ in anchors}
                if any(max(abs(r - s), abs(c1 - d)) <= 1
                       for r, c1 in points for s, d in points if (r, c1) != (s, d)):
                    continue
                result.append({"box": (top, bottom, left, right), "fill": fill,
                               "bands": bands, "anchors": anchors, "keys": keys})
    return result


def _read_bank(g, board):
    bg = int(g[0, 0])
    top, bottom, left, right = board["box"]
    mask = g != bg
    mask = mask.copy()
    mask[top:bottom + 1, left:right + 1] = False
    components = _components(zip(*np.nonzero(mask)))
    if not components:
        raise ValueError("missing reference bank")
    templates, raw, repetitions, body_colors, poses = {}, {}, Counter(), set(), {}
    for component in components:
        key_cells = [p for p in component if int(g[p]) in board["keys"]]
        if len(key_cells) != 1:
            raise ValueError("each reference needs exactly one keyed anchor")
        ar, ac = key_cells[0]
        key = int(g[ar, ac])
        cells = {(int(r - ar), int(c - ac)): int(g[r, c]) for r, c in component}
        labels = {p: v for p, v in cells.items() if v in board["bands"]}
        if len(labels) != 2 or len(set(labels.values())) != 2:
            raise ValueError("two distinct compass terminals are required")
        directions = [_direction(p) for p in labels]
        if (directions[0] - directions[1]) % 2 != 1:
            raise ValueError("compass arms must be perpendicular")
        decor = {p: v for p, v in cells.items() if p != (0, 0) and p not in labels}
        colors = set(decor.values())
        if len(colors) != 1 or len(decor) < 3:
            raise ValueError("one common decoration color and a complete motif are required")
        body_colors.update(colors)
        body_color = next(iter(colors))
        if body_color in board["keys"] | set(board["bands"]) | {bg, board["fill"]}:
            raise ValueError("decoration role is ambiguous")
        for p in labels:
            direction = DIRS[_direction(p)]
            length = abs(p[0]) + abs(p[1])
            if length < 2 or any(cells.get((i * direction[0], i * direction[1])) != body_color
                                 for i in range(1, length)):
                raise ValueError("compass terminals need visible connecting arms")
        choices = []
        for t in range(8):
            transformed = _transform(cells, t)
            if all(board["bands"][_direction(p)] == v for p, v in transformed.items()
                   if v in board["bands"]):
                choices.append((t, {p: v for p, v in transformed.items() if v not in board["bands"]}))
        if len(choices) != 1:
            raise ValueError("compass does not determine one dihedral pose")
        t, template = choices[0]
        if key in templates and (_signature(templates[key]) != _signature(template)
                                 or _signature(raw[key]) != _signature({p: v for p, v in cells.items()
                                                                      if v not in board["bands"]})):
            raise ValueError("conflicting repeated references")
        templates[key] = template
        raw[key] = {p: v for p, v in cells.items() if v not in board["bands"]}
        poses[key] = t
        repetitions[key] += 1
    if set(templates) != board["keys"] or len(body_colors) != 1:
        raise ValueError("the bank must define every key using a common decoration color")
    return templates, raw, repetitions, next(iter(body_colors)), poses


def _render(board_grid, anchors, templates, decor, parity=True):
    h, w = board_grid.shape
    counts = np.zeros((h, w), dtype=int)
    protected = {(r, c) for r, c, _ in anchors}
    for ar, ac, key in anchors:
        for (dr, dc), color in templates[key].items():
            if (dr, dc) == (0, 0):
                continue
            r, c = ar + dr, ac + dc
            if not (1 <= r < h - 1 and 1 <= c < w - 1) or (r, c) in protected:
                raise ValueError("stamp crosses a side or an anchor")
            if color != decor:
                raise ValueError("inconsistent decoration color")
            counts[r, c] += 1
    out = board_grid.copy()
    out[(counts % 2 == 1) if parity else (counts > 0)] = decor
    return out, counts


def _analyze(input_grid):
    g = _grid(input_grid)
    interpretations = []
    for board in _board_candidates(g):
        try:
            templates, raw, reps, decor, poses = _read_bank(g, board)
            t, b, l, r = board["box"]
            crop = g[t:b + 1, l:r + 1].copy()
            out, coverage = _render(crop, board["anchors"], templates, decor)
            info = {**board, "templates": templates, "raw": raw, "repetitions": reps,
                    "decor": decor, "poses": poses, "crop": crop, "coverage": coverage}
            interpretations.append((out, info))
        except ValueError:
            continue
    if len(interpretations) != 1:
        raise ValueError("input must determine exactly one board and dictionary")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Orient the visible reference masks and weave them at the visible keys."""
    return _analyze(input_grid)[0]


def _make_template(rng, key, decor, bands, radius):
    for _ in range(100):
        first = int(rng.integers(4))
        directions = (first, (first + int(rng.choice([-1, 1]))) % 4)
        core, labels = {(0, 0)}, {}
        for d in directions:
            dr, dc = DIRS[d]
            length = int(rng.integers(2, radius + 2))
            core.update((i * dr, i * dc) for i in range(1, length))
            labels[length * dr, length * dc] = int(bands[d])
        size = int(rng.integers(max(5, len(core) + 1), min(22, (2 * radius + 1) ** 2 - 1) + 1))
        while len(core) < size:
            frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in DIRS
                               if abs(r + dr) <= radius and abs(c + dc) <= radius}
                              - core - set(labels))
            if not frontier:
                break
            core.add(frontier[int(rng.integers(len(frontier)))])
        body = {p: int(key if p == (0, 0) else decor) for p in core}
        r0, r1, c0, c1 = _bounds(body)
        # Gaps expose transparency; eight distinct masks make every pose visible.
        if len(body) == (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if len({_signature(_transform(body, t)) for t in range(8)}) != 8:
            continue
        return {**body, **labels}, body
    # An explicit procedural-growth failure is preferable to a hidden rule change.
    raise RuntimeError("unable to grow an asymmetric reference mask")


def _shelves(references):
    result = []
    for limit in range(max(_bounds(c)[3] - _bounds(c)[2] + 1 for c in references), 28):
        row, col, shelf_h, maximum_w, places = 0, 0, 0, 0, []
        for cells in references:
            r0, r1, c0, c1 = _bounds(cells)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if col and col + w > limit:
                row, col, shelf_h = row + shelf_h + 1, 0, 0
            places.append((row - r0, col - c0))
            maximum_w = max(maximum_w, col + w)
            col += w + 1
            shelf_h = max(shelf_h, h)
        bh = row + shelf_h
        if bh <= 27:
            candidate = (bh, maximum_w, tuple(places))
            if candidate not in result:
                result.append(candidate)
    return result


def _anchor_options(body, shape, raw=None):
    bounds = _bounds({**body, **(raw or {})})
    r0, r1, c0, c1 = bounds
    h, w = shape
    return [(r, c) for r in range(max(2, 1 - r0), min(h - 2, h - 1 - r1))
            for c in range(max(2, 1 - c0), min(w - 2, w - 1 - c1))]


def _place_anchors(rng, templates, raw, ids, shape, teach):
    options = {k: _anchor_options(body, shape, raw[k] if teach else None)
               for k, body in templates.items()}
    for _ in range(35):
        anchors, occupied, coverage = [], set(), Counter()
        for index, key in enumerate(ids):
            candidates = options[key].copy()
            rng.shuffle(candidates)
            want_overlap = index == 1 or (index > 1 and rng.random() < .6)
            best = None
            for ar, ac in candidates:
                if any(max(abs(ar - r), abs(ac - c)) <= 1 for r, c, _ in anchors):
                    continue
                mask = {(ar + r, ac + c) for r, c in templates[key] if (r, c) != (0, 0)}
                if (ar, ac) in occupied or any((r, c) in mask for r, c, _ in anchors):
                    continue
                overlap = bool(mask & occupied)
                if best is None or overlap == want_overlap:
                    best = (ar, ac, mask)
                if overlap == want_overlap:
                    break
            if best is None:
                break
            ar, ac, mask = best
            anchors.append((ar, ac, int(key)))
            occupied.update(mask)
            coverage.update(mask)
        else:
            per_key = {}
            for ar, ac, key in anchors:
                for r, c in templates[key]:
                    if (r, c) != (0, 0):
                        per_key.setdefault((ar + r, ac + c), set()).add(key)
            mixed_cancel = any(n == 2 and len(per_key[p]) >= 2 for p, n in coverage.items())
            if (any(n % 2 == 0 for n in coverage.values())
                    and any(n == 1 for n in coverage.values())
                    and (teach != 2 or (2 in coverage.values() and 3 in coverage.values() and mixed_cancel))):
                return anchors
    return None


def _construct(rng, ntypes, target_count, teach, layout):
    palette = [int(c) for c in rng.permutation(10)]
    bg, fill = palette[:2]
    if not teach and rng.random() < .3:
        fill = bg
    bands, decor, keys = palette[2:6], palette[6], palette[7:7 + ntypes]
    templates, raw, references = {}, {}, []
    for i, key in enumerate(keys):
        radius = int(rng.choice([1, 2, 3], p=[.20, .65, .15]))
        full, body = _make_template(rng, key, decor, bands, radius)
        if any(_signature(body) == _signature(t) for t in templates.values()):
            return None
        pose = int(rng.integers(1, 4) if teach == 1 else rng.integers(4, 8)
                   if teach == 2 else rng.integers(8))
        source = _transform(full, pose)
        templates[key], raw[key] = body, _transform(body, pose)
        repeats = 2 if teach == 1 else int(rng.choice([1, 2], p=[.8, .2]))
        references.extend(source.copy() for _ in range(repeats))
    rng.shuffle(references)
    axis = layout
    options = []
    for bh, bw, places in _shelves(references):
        for h in range(10, 26):
            for w in range(10, 26):
                required = (bh + h + 3, max(bw, w) + 2) if axis == 0 else (max(bh, h) + 2, bw + w + 3)
                if max(required) <= 30:
                    options.append((bh, bw, places, h, w, required))
    if not options:
        return None
    rng.shuffle(options)
    # Try several geometries before regrowing masks. Packing is always bounded.
    ids = keys + [keys[0]] + [int(rng.choice(keys)) for _ in range(target_count - ntypes - 1)]
    if teach == 2:
        ids = [keys[0], keys[1]] + ids[2:]
    for bh, bw, places, h, w, required in options[:12]:
        anchors = _place_anchors(rng, templates, raw, ids, (h, w), teach)
        if anchors is not None:
            break
    else:
        return None
    shape = [int(rng.integers(n, min(31, n + 4))) for n in required]
    gap = int(rng.integers(1, min(3, shape[axis] - (bh + h if axis == 0 else bw + w) - 2) + 1))
    bank_shape, board_shape = (bh, bw), (h, w)
    bank_origin, board_origin = [0, 0], [0, 0]
    first_bank = bool(rng.integers(2))
    first, second = (bank_origin, board_origin) if first_bank else (board_origin, bank_origin)
    first_size = bank_shape[axis] if first_bank else board_shape[axis]
    start = int(rng.integers(1, shape[axis] - bank_shape[axis] - board_shape[axis] - gap))
    first[axis], second[axis] = start, start + first_size + gap
    other = 1 - axis
    bank_origin[other] = int(rng.integers(1, shape[other] - bank_shape[other]))
    board_origin[other] = int(rng.integers(1, shape[other] - board_shape[other]))
    g = np.full(shape, bg, dtype=int)
    tr, tc = board_origin
    board = g[tr:tr + h, tc:tc + w]
    board[:] = fill
    board[0, 1:-1], board[-1, 1:-1] = bands[0], bands[2]
    board[1:-1, 0], board[1:-1, -1] = bands[3], bands[1]
    for p in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        board[p] = bg
    for r, c, key in anchors:
        board[r, c] = key
    for cells, (ar, ac) in zip(references, places):
        for (r, c), value in cells.items():
            g[bank_origin[0] + ar + r, bank_origin[1] + ac + c] = value
    out, _ = _render(board, anchors, templates, decor)
    if teach:
        try:
            wrong, _ = _render(board, anchors, raw, decor)
        except ValueError:
            return None
        if np.array_equal(wrong, out):
            return None
        if teach == 2:
            common = {k: {p: (k if p == (0, 0) else v) for p, v in templates[keys[0]].items()}
                      for k in keys}
            try:
                alternative, _ = _render(board, anchors, common, decor)
                if np.array_equal(alternative, out):
                    return None
            except ValueError:
                # Prefer a meaningful, fitting wrong-key comparison.
                return None
    try:
        predicted, _ = _analyze(g)
    except ValueError:
        return None
    if not np.array_equal(predicted, out):
        return None
    return {"input": g, "output": out}


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    type_shift = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teach = index + 1 if split == "train" and index < 2 else 0
            ntypes = teach if teach else (3 if split == "test" and type_shift else int(rng.integers(1, 3)))
            total = (int(rng.integers(3, 6)) if teach == 1 else int(rng.integers(4, 7)) if teach == 2
                     else int(rng.integers(4, 8)) if split == "test" and type_shift
                     else int(rng.integers(7, 10)) if split == "test" else int(rng.integers(3, 7)))
            layout = int(rng.integers(2))
            for _ in range(150):
                pair = _construct(rng, ntypes, total, teach, layout)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                if teach == 2:
                    previous = _analyze(puzzle["train"][0]["input"])[1]["bands"]
                    current = _analyze(pair["input"])[1]["bands"]
                    if not any(v in previous and previous.index(v) != d for d, v in enumerate(current)):
                        continue
                break
            else:
                raise RuntimeError("bounded compass-weave construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    out, info = _analyze(pair["input"])
    assert np.array_equal(out, pair["output"]), "incorrect compass/parity completion"
    assert np.array_equal(out[0], info["crop"][0]) and np.array_equal(out[-1], info["crop"][-1])
    assert np.array_equal(out[:, 0], info["crop"][:, 0]) and np.array_equal(out[:, -1], info["crop"][:, -1])
    for r, c, key in info["anchors"]:
        assert out[r, c] == key
    coverage = info["coverage"]
    assert np.all(out[coverage > 0] == np.where(coverage[coverage > 0] % 2,
                                              info["decor"], info["fill"]))
    assert np.any(coverage == 1) and np.any((coverage > 0) & (coverage % 2 == 0))
    assert np.any(out == info["decor"])
    for template in info["templates"].values():
        assert len(_components(template)) == 1
        assert len({_signature(_transform(template, t)) for t in range(8)}) == 8
    try:
        wrong, _ = _render(info["crop"], info["anchors"], info["raw"], info["decor"])
        compass_witness = not np.array_equal(wrong, out)
    except ValueError:
        compass_witness = False
    key_witness = False
    for first_key in info["templates"]:
        common = {k: {p: (k if p == (0, 0) else v) for p, v in info["templates"][first_key].items()}
                  for k in info["templates"]}
        try:
            alternative, _ = _render(info["crop"], info["anchors"], common, info["decor"])
            key_witness |= not np.array_equal(alternative, out)
        except ValueError:
            pass
    t, b, l, r = info["box"]
    exterior_points = [(rr, cc) for rr, cc in zip(*np.nonzero(pair["input"] != pair["input"][0, 0]))
                       if not (t <= rr <= b and l <= cc <= r)]
    bank_r0, bank_r1, bank_c0, bank_c1 = _bounds(exterior_points)
    layout = "vertical" if bank_r1 < t or bank_r0 > b else "horizontal"
    per_key = {}
    for ar, ac, key in info["anchors"]:
        for r0, c0 in info["templates"][key]:
            if (r0, c0) != (0, 0):
                per_key.setdefault((ar + r0, ac + c0), set()).add(key)
    mixed_cancel = any(coverage[p] == 2 and len(ks) >= 2 for p, ks in per_key.items())
    return {"input_shape": tuple(pair["input"].shape), "output_shape": tuple(out.shape),
            "keys": len(info["keys"]), "anchors": len(info["anchors"]), "layout": layout,
            "sizes": tuple(len(c) - 1 for c in info["templates"].values()),
            "masks": tuple(tuple(sorted(c)) for c in info["templates"].values()),
            "poses": tuple(info["poses"].values()), "repetitions": tuple(info["repetitions"].values()),
            "coverage_max": int(coverage.max()), "cancelled": int(np.count_nonzero((coverage > 0) & (coverage % 2 == 0))),
            "coverage_levels": tuple(int(n) for n in np.unique(coverage) if n),
            "mixed_cancel": mixed_cancel, "bands": info["bands"],
            "equal_fills": int(pair["input"][0, 0]) == info["fill"],
            "compass_witness": compass_witness, "key_witness": key_witness,
            "transparency_witness": info["fill"] != int(pair["input"][0, 0])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            features = _features({"input": inp, "output": out})
            (training if split == "train" else testing).append(features)
    assert training[0]["keys"] == 1 and training[1]["keys"] == 2
    assert all(f["compass_witness"] and f["transparency_witness"] for f in training[:2])
    assert training[1]["key_witness"], "training must distinguish key-specific templates"
    assert {1, 2, 3} <= set(training[1]["coverage_levels"]), "training must show coverage one, two and three"
    assert training[1]["mixed_cancel"], "training must cancel across different keys"
    assert any(v in training[0]["bands"] and training[0]["bands"].index(v) != d
               for d, v in enumerate(training[1]["bands"])), "side identities must change direction"
    assert any(0 < t < 4 for t in training[0]["poses"])
    assert any(t >= 4 for t in training[1]["poses"])
    assert all(f["keys"] > max(t["keys"] for t in training)
               or f["anchors"] > max(t["anchors"] for t in training) for f in testing)


def self_test():
    """Bounded semantic, interface, equivariance and 42-seed variation checks."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                records.append(_features(pair))
        if seed < 3:
            repeated = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], repeated[s]) for k in ("input", "output"))
            pair = puzzle["test"][0]
            for t in range(8):
                inp = np.fliplr(pair["input"]) if t >= 4 else pair["input"]
                out = np.fliplr(pair["output"]) if t >= 4 else pair["output"]
                assert np.array_equal(solve(np.rot90(inp, t % 4)), np.rot90(out, t % 4))
            recolor = np.random.default_rng(1000 + seed).permutation(10)
            assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
    assert len(signatures) == 42
    assert len({f["input_shape"] for f in records}) >= 35
    assert len({f["output_shape"] for f in records}) >= 35
    assert len({m for f in records for m in f["masks"]}) >= 100
    assert {f["keys"] for f in records} == {1, 2, 3}
    assert {f["layout"] for f in records} == {"vertical", "horizontal"}
    assert set(t for f in records for t in f["poses"]) == set(range(8))
    assert {v for f in records for v in f["repetitions"]} == {1, 2}
    assert {f["equal_fills"] for f in records} == {True, False}
    assert max(f["coverage_max"] for f in records) >= 3
    assert len({n for f in records for n in f["sizes"]}) >= 10
    assert {f["anchors"] for f in records} >= {3, 4, 5, 6, 7, 8, 9}
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(991 + a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 1] = (int(bad["test"][0]["output"][0, 1]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted border accepted")
    # A double-covered cell must be restored to fill, even across key identities.
    broken = generate_puzzle(1)
    _, info = _analyze(broken["train"][1]["input"])
    r, c = map(int, np.argwhere(info["coverage"] == 2)[0])
    broken["train"][1]["output"][r, c] = info["decor"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("painted even coverage accepted")
    # Perpendicular input rays cannot name opposite destination sides.
    original = generate_puzzle(2)["train"][0]["input"]
    _, info = _analyze(original)
    t, b, l, r = info["box"]
    outside = original != original[0, 0]
    outside[t:b + 1, l:r + 1] = False
    component = _components(zip(*np.nonzero(outside)))[0]
    labels = [p for p in component if int(original[p]) in info["bands"]]
    malformed = original.copy()
    other_side = info["bands"].index(int(original[labels[1]]))
    malformed[labels[0]] = info["bands"][(other_side + 2) % 4]
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent compass accepted")


if __name__ == "__main__":
    self_test()
    print("Compass weave self-test passed across 42 seeds.")
